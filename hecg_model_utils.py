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
import ecg_vit2 as vits1k

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


def get_vit1k(pretrained_weights, arch='vit1k_xs', device=torch.device('cuda:1')):
    r"""
    Builds ViT-1k Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-1k Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model1k (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model1k = vits1k.__dict__[arch](num_classes=0)
    for p in model1k.parameters():
        p.requires_grad = False
    model1k.eval()
    model1k.to(device)

    if os.path.isfile(pretrained_weights):
        with torch.serialization.safe_globals([multiarray.scalar, np.dtype, np.dtypes.Float64DType, argparse.Namespace]):
            state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model1k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model1k


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
def tensorbatch2im(input_signal, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy series array.

    Args:
        - input_signal (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array

    Returns:
        - signal_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_signal, np.ndarray):
        signal_numpy = input_signal.cpu().float().numpy()  # convert it into a numpy array
        signal_numpy = np.transpose(signal_numpy, (0, 2, 3, 1)) #e.g.(5,12,1,200)=>(5,1,200,12)
    else:  # if it is a numpy array, do nothing
        signal_numpy = input_signal
    return signal_numpy.astype(imtype)
