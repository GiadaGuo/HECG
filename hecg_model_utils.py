### Dependencies
# Base Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys

# LinAlg / Stats / Plotting Dependencies
import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from scipy.stats import rankdata

from tqdm import tqdm
import webdataset as wds

# Torch Dependencies
import torch
import torch.multiprocessing


from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

from numpy.core import multiarray
torch.serialization.add_safe_globals([multiarray.scalar,np.dtype,np.dtypes.Float64DType,argparse.Namespace])

# Local Dependencies
import ecg_vit1 as vits
import ecg_vit2 as vits4k

def get_vit200(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-200 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-200 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model200 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher' #通过预训练最终使用教师模型
    device = torch.device("cpu")
    #根据变量arch的值(arch='vit_small')，动态地从vits模块或类中获取对应的模型类或函数，然后创建这个模型的一个实例
    model200 = vits.__dict__[arch](slice_len=40, num_classes=0)
    for p in model200.parameters():
        p.requires_grad = False
    model200.eval()
    model200.to(device)

    if os.path.isfile(pretrained_weights):
        with torch.serialization.safe_globals([multiarray.scalar,np.dtype,np.dtypes.Float64DType,argparse.Namespace]):
            state_dict = torch.load(pretrained_weights, map_location="cpu",weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model200.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model200


def get_vit4k(pretrained_weights, arch='vit4k_xs', device=torch.device('cuda:1')):
    r"""
    Builds ViT-4K Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-4K Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model4k = vits4k.__dict__[arch](num_classes=0)
    for p in model4k.parameters():
        p.requires_grad = False
    model4k.eval()
    model4k.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model4k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model4k


def eval_transforms(signal):
    """
    返回一个用于numpy数组预处理的转换管道
    后续可添加normalize操作，目前不考虑
    """

    #转置为 (12, 200)=> 加一个伪维度，变成 (12, 1, 200)
    eval_t = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(1)

    return eval_t


# def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):
# 	"""
# 	Rolls an image tensor batch (batch of [256 x 256] images) into a [W x H] Pil.Image object.
#
# 	Args:
# 		batch (torch.Tensor): [B x 3 x 256 x 256] image tensor batch.
#
# 	Return:
# 		Image.PIL: [W x H X 3] Image.
# 	"""
# 	batch = batch.reshape(w, h, 3, patch_size, patch_size)
# 	img = rearrange(batch, 'p1 p2 c w h-> c (p1 w) (p2 h)').unsqueeze(dim=0)
# 	return Image.fromarray(tensorbatch2im(img)[0])
#
#
# def tensorbatch2im(input_image, imtype=np.uint8):
#     r""""
#     Converts a Tensor array into a numpy image array.
#
#     Args:
#         - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
#         - imtype (type): the desired type of the converted numpy array
#
#     Returns:
#         - image_numpy (np.array): (B, W, H, C) Numpy Array.
#     """
#     if not isinstance(input_image, np.ndarray):
#         image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
#         #if image_numpy.shape[0] == 1:  # grayscale to RGB
#         #    image_numpy = np.tile(image_numpy, (3, 1, 1))
#         image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#     else:  # if it is a numpy array, do nothing
#         image_numpy = input_image
#     return image_numpy.astype(imtype)
