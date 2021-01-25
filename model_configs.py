from modules.unet import UNet
from modules.percep_nets import ResidualsNetDown, ResidualsNetUp
from task_configs import get_task
from utils import *

model_types = {
    'unet_based':{
        'normal': {
            'down': (lambda: UNet(in_channels=3, downsample=3), f"{MODELS_DIR}/normal_down.pth"),
            'up' : (lambda: UNet(out_channels=3, downsample=3), f"{MODELS_DIR}/normal_up.pth"),
        },
        'sobel_edges': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
            'up': (lambda: UNet(out_channels=1, downsample=3), f"{MODELS_DIR}/sobel_edges_up.pth"),
        },
        'reshading': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/reshading_down.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/reshading_up.pth"),
        },
        'keypoints2d': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
        },
        'keypoints3d': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
            'up' : (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
        },
        'depth_zbuffer': {
            'down': (lambda: UNet(in_channels=1, downsample=3), f"{MODELS_DIR}/depth_zbuffer_down.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/depth_zbuffer_up.pth"),
        },
        'principal_curvuture': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/principal_curvature_down.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
        },
        'edge_occlusion': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
            'up': (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
        },
        'rgb': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/rgb_down.pth"),
            'up': (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/rgb_up.pth"),
        },
    },
    'resnet_based':{
        'normal': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/normal_down.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/normal_up.pth"),
        },
        'sobel_edges': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/sobel_edges_up.pth"),
        },
        'reshading': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/reshading_down.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/reshading_up.pth"),
        },
        'keypoints2d': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
        },
        'keypoints3d': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
            'up' : (lambda: ResidualsNetUp(in_channels=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
        },
        'depth_zbuffer': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/depth_zbuffer_down.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/depth_zbuffer_up.pth"),
        },
        'principal_curvuture': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/principal_curvature_down.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
        },
        'edge_occlusion': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
        },
        'rgb': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/rgb_down.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/rgb_up.pth"),
        },
    },
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=5),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=6, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
    ('rgb', 'normal'): lambda : UNet(downsample=6),
    ('rgb', 'keypoints2d'): lambda : UNet(downsample=3, out_channels=1),
}


def get_model(src_task, dest_task):

    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task, dest_task = get_task(src_task), get_task(dest_task)

    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ImageTask):
        return UNet(downsample=3, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ClassTask):
        return ResNet(in_channels=src_task.shape[0], out_channels=dest_task.classes)

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, PointInfoTask):
        return ResNet(out_channels=dest_task.out_channels)

    return None


def get_model_LS(task, type_name='unet_based'):
    task_name = task
    
    if isinstance(task, ImageTask):
        task = task.name
    
    assert isinstance(task_name, str), "Name of task is not a string"
    assert task_name in model_types[type_name], "Name of task doesn't have relative model"
    models = model_types[type_name][task_name]
    return models