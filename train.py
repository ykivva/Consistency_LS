'''
  Name: train.py
  Desc: Executes training of a network with the consistency framework.

    Here are some options that may be specified for any model. If they have a
    default value, it is given at the end of the description in parens.

        Data pipeline:
            Data locations:
                'train_buildings': A list of the folders containing the training data. This
                    is defined in configs/split.txt.
                'val_buildings': As above, but for validation data.
                'data_dirs': The folder that all the data is stored in. This may just be
                    something like '/', and then all filenames in 'train_filenames' will
                    give paths relative to 'dataset_dir'. For example, if 'dataset_dir'='/',
                    then train_filenames might have entries like 'path/to/data/img_01.png'.
                    This is defiled in utils.py.

        Logging:
            'results_dir': An absolute path to where checkpoints are saved. This is
                defined in utils.py.

        Training:
            'batch_size': The size of each batch. (64)
            'num_epochs': The maximum number of epochs to train for. (800)
            'energy_config': {multiperceptual_targettask} The paths taken to compute the losses.
            'k': Number of perceptual loss chosen.
            'data_aug': {True, False} If data augmentation shuold be used during training.
                See TrainTaskDataset class in datasets.py for the types of data augmentation
                used. (False)

        Optimization:
            'initial_learning_rate': The initial learning rate to use for the model. (3e-5)


  Usage:
    python -m train perceptual_normal --batch-size 64 --k 8 --max-epochs 100
'''

import torch
import torch.nn as nn

from utils import *
from energy import get_energy_loss
from graph import TaskGraph
from logger import Logger, VisdomLogger
from datasets import load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask

from fire import Fire

import pdb

def main(
    job_config="jobinfo.txt", models_dir="models",
    fast=False, batch_size=None,
    subset_size=None, max_epochs=500, dataaug=False, **kwargs,
):
    loss_config, loss_mode, model_class = None, None, None
    experiment, base_dir = None, None
    current_dir = os.path.dirname(__file__)
    job_config = os.path.normpath(
        os.path.join(
            os.path.join(current_dir, "config"), job_config
            )
        )
    if os.path.isfile(job_config):
        with open(job_config) as config_file:
            out = config_file.read().strip().split(',\n')
            loss_config, loss_mode, model_class, experiment, base_dir = out
    loss_config = loss_config or LOSS_CONFIG
    loss_mode = loss_mode or LOSS_MODE
    model_class = model_class or MODEL_CLASS
    base_dir = base_dir or BASE_DIR

    base_dir = os.path.normpath(os.path.join(current_dir, base_dir))
    experiment = experiment or EXPERIMENT
    job = "_".join(experiment.split("_")[0:-1])

    models_dir = os.path.join(base_dir, models_dir)
    results_dir = f"{base_dir}/results/results_{experiment}"
    results_dir_models = f"{base_dir}/results/results_{experiment}/models"

    # CONFIG
    batch_size = batch_size or (4 if fast else 64)
    energy_loss = get_energy_loss(config=loss_config, loss_mode=loss_mode, **kwargs)

    # DATA LOADING
    train_dataset, val_dataset, train_step, val_step = load_train_val(
        energy_loss.get_tasks("train"),
        batch_size=batch_size,
        fast=fast,
        subset_size=subset_size,
        dataaug=dataaug,
    )

    if fast:
        train_dataset = val_dataset
        train_step, val_step = 2,2

    train = RealityTask(
        "train", train_dataset, batch_size=batch_size, shuffle=True
    )
    val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
    
    test_set = load_test(
        energy_loss.get_tasks("test"),
        buildings=['almena', 'albertville','espanola']
    )
    ood_set = load_ood(energy_loss.get_tasks("ood"))
    test = RealityTask.from_static(
        "test", test_set, energy_loss.get_tasks("test")
    )
    ood = RealityTask.from_static("ood", ood_set, [tasks.rgb,])
    realities = [train, val, test, ood]

    # GRAPH
    graph = TaskGraph(
        tasks=energy_loss.tasks + realities,
        tasks_in=energy_loss.tasks_in,
        tasks_out=energy_loss.tasks_out,
        pretrained=True, models_dir=models_dir,
        freeze_list=energy_loss.freeze_list,
        direct_edges=energy_loss.direct_edges,
        model_class=model_class
    )
    graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
    
    # LOGGING
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_models, exist_ok=True)
    logger = VisdomLogger("train", env=job, port=PORT, server=SERVER)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    logger.add_hook(
        lambda _, __: graph.save(f"{results_dir}/graph.pth", results_dir_models),
        feature="epoch",
        freq=1
    )
    energy_loss.logger_hooks(logger)
    energy_loss.plot_paths(graph, logger, realities, prefix="start")

    # BASELINE
    graph.eval()
    with torch.no_grad():
        for _ in range(0, val_step*4):
            val_loss, _ = energy_loss(
                graph, realities=[val]
            )
            val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)
            
        for _ in range(0, train_step*4):
            train_loss, _ = energy_loss(
                graph, realities=[train]
            )
            train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
            train.step()
            logger.update("loss", train_loss)
    energy_loss.logger_update(logger)

    # TRAINING
    for epochs in range(0, max_epochs):
        logger.update("epoch", epochs)
        energy_loss.plot_paths(graph, logger, realities, prefix="finish")
        
        graph.train()
        for _ in range(0, train_step):
            train_loss, grad_mse_coeff = energy_loss(
                graph, realities=[train],
                compute_grad_ratio=True
            )
            graph.step(train_loss, losses=energy_loss.losses, paths=energy_loss.paths)
            train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
            train.step()
            logger.update("loss", train_loss)
            del train_loss
        print(grad_mse_coeff)

        graph.eval()
        for _ in range(0, val_step):
            with torch.no_grad():
                val_loss, _ = energy_loss(graph, realities=[val])
                val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)
            
        energy_loss.logger_update(logger)
        logger.step()
    
            
if __name__ == "__main__":
    Fire(main)
