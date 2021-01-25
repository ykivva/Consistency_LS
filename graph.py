import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

import task_configs

from utils import *
from models import TrainableModel, WrapperModel, DataParallelModel
from datasets import TaskDataset
from task_configs import get_task, task_map, RealityTask, ImageTask, Task
from transfers import RealityTransfer, Transfer
from model_configs import model_types

#from modules.gan_dis import GanDisNet

import pdb

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks, tasks_in={}, tasks_out={},
        pretrained=True, finetuned=False,
        freeze_list=[], direct_edges={}, lazy=False,
        model_class='resnet_based'
    ):
        super().__init__()
        self.tasks = tasks
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.tasks_in, self.tasks_out = tasks_in, tasks_out
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges_in, self.edges_out, = {}, {}
        self.direct_edges = direct_edges
        self.freeze_list = freeze_list
        self.edge_map = {}
        print('Creating graph with tasks:', self.tasks)
        self.params = {}
        transfer_models = model_types[model_class]
        
        for task in self.tasks_out.get("edges", None):
            key = str((task.name, "LS"))
            model_type, path = transfer_models.get(task.name, {})["down"]
            if not os.path.isfile(path):
                path = None
            transfer = Transfer(
                task, task_configs.tasks.LS,
                model_type=model_type, path=path
            )
                
            transfer.freezed = task in self.tasks_out.get("freeze")
            self.edges_out[task.name] = transfer
            self.edge_map[key] = transfer
            
            if transfer.freezed:
                transfer.set_requires_grad(False)
            else:
                self.params[key] = transfer
            
            try:
                if not lazy:
                    transfer.load_model()
            except Exception as e:
                print(e)
                IPython.embed()
        
        for task in self.tasks_in.get("edges", None):
            key = str(("LS", task.name))
            model_type, path = transfer_models.get(task.name, {})["up"]
            if not os.path.isfile(path):
                path = None
            transfer = Transfer(
                task_configs.tasks.LS, task,
                model_type=model_type, path=path
            )
            transfer.freezed = task in self.tasks_in.get("freeze")
            self.edges_in[task.name] = transfer
            self.edge_map[key] = transfer
            
            if transfer.freezed: 
                transfer.set_requires_grad(False)
            else:
                self.params[key] = transfer
            
            try:
                if not lazy:
                    transfer.load_model()
            except Exception as e:
                print(e)
                IPython.embed()
        
        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = str((src_task.name, dest_task.name))
            transfer = None
            if src_task==dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            if src_task==task_configs.tasks.LS or dest_task==task_configs.tasks.LS:
                continue
            if isinstance(src_task, RealityTask):
                transfer = RealityTransfer(src_task, dest_task)
                self.edge_map[key] = transfer
            elif key in self.direct_edges:
                transfer = Transfer(src_task, dest_task, pretrained=pretrained, finetuned=finetuned)
                transfer.freezed = key in self.freeze_list
                
                if transfer.model_type is None:
                    continue
                if not transfer.freezed:
                    self.params[key] = transfer
                else:
                    print("Setting link: " + str(key) + " not trainable.")
                    transfer.set_requires_grad(False)
                
                try:
                    if not lazy: transfer.load_model()
                except Exception as e:
                    print(e)
                    IPython.embed()
            else: continue

            self.edge_map[key] = transfer
        
        self.params = nn.ModuleDict(self.params)
    
    def edge(self, src_task, dest_task):
        key = str((src_task.name, dest_task.name))
        isdirect = key in self.direct_edges
        if isinstance(src_task, ImageTask) and isinstance(dest_task, ImageTask) and not isdirect:
            return lambda x: self.edges_in[dest_task.name](self.edges_out[src_task.name](x))
        return self.edge_map[key]

    def sample_path(self, path, reality, use_cache=False, cache={}):
        path = [reality] + path
        x = None
        for i in range(1, len(path)):
            try:
                model = self.edge(path[i-1], path[i])
                x = cache.get(tuple(path[0:(i+1)]), model(x))

            except KeyError:
                return None
            except Exception as e:
                print(e)
                IPython.embed()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        
        return x
    
    def step(self, loss, train=True, losses=None, paths=None):
        self.zero_grad()
        self.optimizer.zero_grad()
        self.train(train)
        self.zero_grad()
        self.optimizer.zero_grad()
        
        if losses is not None:
            for loss_name in loss:
                loss_config = list(losses[loss_name].values())[0]
                path_names, paths_grads = loss_config
                for path_name, path_grads in zip(path_names, paths_grads):
                    self.set_requires_grad_path(paths[path_name], path_grads)
                loss[loss_name].backward(retain_graph=True)
        else:
            loss.backward()
        
        self.optimizer.step()
        if losses is not None:
            for loss_name in loss:
                loss_config = list(losses[loss_name].values())[0]
                path_names, paths_grads = loss_config
                for path_name, path_grads in zip(path_names, paths_grads):
                    path_grads = [True for _ in path_grads]
                    path_grads[0] = None
                    self.set_requires_grad_path(paths[path_name], path_grads)
        
        self.zero_grad()
        self.optimizer.zero_grad()
    
    def set_requires_grad_path(self, path, path_grads):
        for i in range(1, len(path)):
            key = str((path[i-1].name, path[i].name))
            if key in self.edge_map:
                model = self.edge_map[key]
                model.set_requires_grad(path_grads[i])
            else:
                key1 = str((path[i-1].name, "LS"))
                key2 = str(("LS", path[i].name))
                model1 = self.edge_map[key1]
                model2 = self.edge_map[key2]
                model1.set_requires_grad(path_grads[i])
                model2.set_requires_grad(path_grads[i])
                
    def get_edge_parameters(self, src_name, dest_name):
        params = []
        key = str((src_name, dest_name))
        if key in self.edge_map:
            model = self.edge_map[key]
            params += list(model.model.parameters())
        else:
            key1 = str((src_name, "LS"))
            key2 = str(("LS", dest_name))
            model1 = self.edge_map[key1]
            model2 = self.edge_map[key2]
            model1.set_requires_grad(path_grads[i])
            model2.set_requires_grad(path_grads[i])
            params += list(model1.model.parameters())
            params += list(model2.model.parameters())
        return params
                
    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            checkpoint = {
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not isinstance(model, RealityTransfer)
            }
            torch.save(checkpoint, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                if isinstance(model, Transfer):
                    model.model.save(f"{weights_dir}/{model.name}.pth")
                    
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")
    
    def load_weights(self, weights_file=None):
        loaded_something = False
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map:
                loaded_something = True
                self.edge_map[key].load_model()
                self.edge_map[key].load_state_dict(state_dict)
        if not loaded_something:
            raise RuntimeError(f"No edges loaded from file: {weights_file}")