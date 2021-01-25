from modules.unet import UNet
from modules.percep_nets import ResidualsNetDown, ResidualsNetUp
from task_configs import get_task
from utils import *

model_types = {
    'unet_based':{
        'normal': {
            'down': (lambda: UNet(in_channels=3, downsample=3), f"{MODELS_DIR}/LS2normal.pth"),
            'up' : (lambda: UNet(out_channels=3, downsample=3), f"{MODELS_DIR}/normal2LS.pth"),
        },
        'sobel_edges': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/sobel_edges2LS.pth"),
            'up': (lambda: UNet(out_channels=1, downsample=3), f"{MODELS_DIR}/LS2sobel_edges.pth"),
        },
        'reshading': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/reshading2LS.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/LS2reshading.pth"),
        },
        'keypoints2d': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/keypoints2d2LS.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/LS2keypoints2d.pth"),
        },
        'keypoints3d': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/keypoints3d2LS.pth"),
            'up' : (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/LS2keypoints3d.pth"),
        },
        'depth_zbuffer': {
            'down': (lambda: UNet(in_channels=1, downsample=3), f"{MODELS_DIR}/depth_zbuffer2LS.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/LS2depth_zbuffer.pth"),
        },
        'principal_curvuture': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/principal_curvature2LS.pth"),
            'up' : (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/LS2principal_curvature.pth"),
        },
        'edge_occlusion': {
            'down': (lambda: UNet(downsample=3, in_channels=1), f"{MODELS_DIR}/edge_occlusion2LS.pth"),
            'up': (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/LS2edge_occlusion.pth"),
        },
        'rgb': {
            'down': (lambda: UNet(downsample=3, in_channels=3), f"{MODELS_DIR}/rgb2LS.pth"),
            'up': (lambda: UNet(downsample=3, out_channels=3), f"{MODELS_DIR}/LS2rgb.pth"),
        },
    },
    'resnet_based':{
        'normal': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/normal2LS.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/LS2normal.pth"),
        },
        'sobel_edges': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/sobel_edges2LS.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/LS2sobel_edges.pth"),
        },
        'reshading': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/reshading2LS.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/LS2reshading.pth"),
        },
        'keypoints2d': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/keypoints2d2LS.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/LS2keypoints2d.pth"),
        },
        'keypoints3d': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/keypoints3d2LS.pth"),
            'up' : (lambda: ResidualsNetUp(in_channels=1), f"{MODELS_DIR}/LS2keypoints3d.pth"),
        },
        'depth_zbuffer': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/depth_zbuffer2LS.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/LS2depth_zbuffer.pth"),
        },
        'principal_curvuture': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/principal_curvature2LS.pth"),
            'up' : (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/LS2principal_curvature.pth"),
        },
        'edge_occlusion': {
            'down': (lambda: ResidualsNetDown(in_channels=1), f"{MODELS_DIR}/edge_occlusion2LS.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=1), f"{MODELS_DIR}/LS2edge_occlusion.pth"),
        },
        'rgb': {
            'down': (lambda: ResidualsNetDown(in_channels=3), f"{MODELS_DIR}/rgb2LS.pth"),
            'up': (lambda: ResidualsNetUp(out_channels=3), f"{MODELS_DIR}/LS2rgb.pth"),
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