### Dependencies
# Base Dependencies
import os
import pickle
import sys

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Local Dependencies
import HECG.ecg_vit1 as vits
import HECG.ecg_vit2 as vits1k
# from HECG_heatmap_utils import *
from HECG.hecg_model_utils import *


class Extract_5K(torch.nn.Module):
	"""
	HECG Model (ViT-1K) for encoding ECG series (with 200 slice tokens), with 200 slice tokens
	encoded via ViT-200 using 40 slice tokens.
	"""
	def __init__(self,
				 size_arg = "small",

		model200_path: str = '../ckpt/vit200_dino_0625.pth',
		model1k_path: str = '../ckpt/vit1k_dino_0625.pth',
		device200=torch.device('cuda:0'), 
		device1k=torch.device('cuda:1')):

		super().__init__()

		#避免没有多个gpu的情况
		num_gpus = torch.cuda.device_count()  # 获取 GPU 数量

		if num_gpus > 1:
			device200 = torch.device('cuda:0')
			device1k = torch.device('cuda:1')
		else:
			device200 = device1k = torch.device('cuda:0')  # 只有 1 块 GPU，就都放在同一块上

		self.model200 = get_vit200(pretrained_weights=model200_path).to(device200)
		self.model1k = get_vit1k(pretrained_weights=model1k_path).to(device1k)
		self.device200 = device200
		self.device1k = device1k

		self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
		self.size = self.size_dict_path[size_arg]

		self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25))
		self.global_transformer = nn.TransformerEncoder(  # WSI层次不用VIT
			nn.TransformerEncoderLayer(
				d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
			),
			num_layers=2
		)
		self.global_attn_pool = Attn_Net_Gated(L=self.size[1], D=self.size[1], dropout=0.25, n_classes=1)
		self.global_rho = nn.Sequential(*[nn.Linear(self.size[1], self.size[1]), nn.ReLU(), nn.Dropout(0.25)])
	
	def forward(self, x):
		"""
		Forward pass of HECG (given an series tensor x), outputting the [CLS] token from ViT-5k.
		1. x is center-cropped(中间剪裁) such that the W / H is divisible by the slice token size in ViT-1k (e.g. - 200).
		2. x then gets unfolded into a "batch" of [200] series.
		3. A pretrained ViT-200 model extracts the CLS token from each [200] series in the batch.
		4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_200" and height "h_200".)
		5. This feature grid is then used as the input to ViT-1k, extracting [CLS]_1k.
		6. outputting [CLS]_5k
		
		Args:
			- x (torch.Tensor): [B x C x 1 x H'] tensor.
			输入：形状是 [B × C × 1 × H']，即 1 张 12导联采样点H的心电图样本，经过裁剪后的大小是 1 × H'
		
		Return:
			- features_cls1k (torch.Tensor): [B x 192] cls token (d_1k = 192 by default).
			输出：形状是 [B × 192]，表示整个 x 的全局特征，192 维的1个向量
		"""
		batch_200, w_200, h_200 = self.prepare_ser_tensor(x)                    # 1. batch_200：[B x 12 x 1 x H]=>e.g.[B x 12 x 1 x 5000];h_200通常是25
		B = batch_200.shape[0]


		batch_200 = batch_200.unfold(3, 200, 200)           # 2. [B, 12, 1, 25, 200]
		batch_200 = rearrange(batch_200, 'b c w n h -> (b n) c w h')    # 2. [B' x 12 x 1 x 200], where B' = B*(1*w_200*h_200) =B*25

		#vit1:200-40
		##INPUT:(B,C,1,H)  OUTPUT:(B,384)
		batch_200=batch_200.to(self.device200, non_blocking=True)
		features_cls200 = self.model200(batch_200).detach().cpu() # 3. [B' x 384]=[(25*B) x 384], where 384 == dim of ViT-200 [ClS] token.

		#维度变换还原成最开始的2D输入形式 (B,C,W,H)——(B,384, w_200,h_200)，便于下一层输入
		#过程：(B,25,384)->(25,B,384)->(5,1,384,5)->transpose(1,2)->(5,384,1,5)
		#过程：((25*B) x 384)->(B,25,384)->(B,5,384,5)->(B*5,384,1,5)
		features_cls200 = features_cls200.reshape(-1,25,self.size[0]).unfold(1,5,5).reshape(-1, self.size[0], 1, 5) #self.size[0]=384

		#vit2:1000-200
		##INPUT:(B*5,384,1,5) OUTPUT:(B*5,192)
		features_cls200 = features_cls200.to(self.device1k, non_blocking=True)  # 4. [B*5 x 384 x 1 x 5]
		features_cls1k = self.model1k.forward(features_cls200)                  # 5. [B*5 x 192], where 192 == dim of ViT-1k [ClS] token.

		#finetuning
		## 自注意力在每个样本的5个token间做
		features_cls1k = self.global_phi(features_cls1k) #[(B*5) x 192]
		features_cls1k = features_cls1k.reshape(-1, 5, self.size[1]) #[Bx 5 x 192]

		features_cls1k = self.global_transformer(features_cls1k) #input[B x 5 x 192]-->output[B x 5 x 192]，5个token间自注意力

		A_1k, features_cls1k = self.global_attn_pool(features_cls1k) #A1k:attention 权重,维度为 [B,5]; features_cls1k [B x 5 x 192]

		A_1k = F.softmax(A_1k, dim=1) # 沿 N 方向 softmax 得到 attention score
		h_path = torch.matmul(A_1k.unsqueeze(1), features_cls1k).squeeze(1) # [B,1,5]*[B,5,192] -> [B,192]

		features_cls5k = self.global_rho(h_path)


		return features_cls5k #[B,192]
	
	
	def forward_asset_dict(self, x: torch.Tensor):
		"""
		Forward pass of HECG (given an image tensor x), with certain intermediate representations saved in 
		a dictionary (that is to be stored in a H5 file). See walkthrough of how the model works above.
		
		Args:
			- x (torch.Tensor): [1 x C x W' x H'] image tensor.
		
		Return:
			- asset_dict (dict): Dictionary of intermediate feature representations of HECG and other metadata.
				- features_cls200 (np.array): [B x 384] extracted ViT-200 cls tokens
				- features_mean256 (np.array): [1 x 384] mean ViT-200 cls token (exluding non-tissue slices)
				- features_1k (np.array): [1 x 192] extracted ViT-1k cls token.
				- features_1k (np.array): [1 x 576] feature vector (concatenating mean ViT-200 + ViT-1k cls tokens)
	
		"""
		batch_200, w_200, h_200 = self.prepare_ser_tensor(x)
		batch_200 = batch_200.unfold(3, 200, 200)
		batch_200 = rearrange(batch_200, 'b c w n h -> (b n) c w h')
		
		features_cls200 = []
		for mini_bs in range(0, batch_200.shape[0], 5):
			minibatch_200 = batch_200[mini_bs:mini_bs+5].to(self.device200, non_blocking=True)
			features_cls200.append(self.model200(minibatch_200).detach().cpu())

		features_cls200 = torch.vstack(features_cls200)
		features_mean200 = features_cls200.mean(dim=0).unsqueeze(dim=0)

		features_grid200 = features_cls200.reshape(w_200, h_200, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
		features_grid200 = features_grid200.to(self.device1k, non_blocking=True)
		features_cls1k = self.model1k.forward(features_grid200).detach().cpu()
		features_mean200_cls1k = torch.cat([features_mean200, features_cls1k], dim=1)
		
		asset_dict = {
			'features_cls200': features_cls200.numpy(),
			'features_mean256': features_mean200.numpy(),
			'features_cls1k': features_cls1k.numpy(),
			'features_mean256_cls1k': features_mean200_cls1k.numpy()
		}
		return asset_dict


	def _get_signal_attention_scores(self, signal):
		r"""
		Forward pass in hierarchical model with attention scores saved.
		
		Args:
		- signal:       ECG series with 1000 sampling points，dim=(B,1000,12)
		- model200 (torch.nn):      200-Level ViT
		- model1k (torch.nn):       1000-Level ViT
		
		Returns:
		- np.array: [5, 1, 200/scale, 12] np.array sequence from the 1000 sampling points series.
		- attention_200 (torch.Tensor): [5, 1, 200/scale, 12] torch.Tensor sequence of attention maps for 200-sized slices.
		- attention_1k (torch.Tensor): [1, 1, 1000/scale, 12] torch.Tensor sequence of attention maps for 1k-sized series.
		"""

		x = eval_transforms(signal) # (B,1000,12)-> (B,12,1,1000)

		batch_200, w_200, h_200 = self.prepare_ser_tensor(x) # batch_200==(B,12,1,1000),w_200==1, h_200==5
		B = batch_200.shape[0]

		batch_200 = batch_200.unfold(3, 200, 200) #(B,12,1,5,200)
		batch_200 = rearrange(batch_200, 'b c w n h -> (b n) c w h') #(B,12,1,5,200)->(B*5,12,1,200)
		batch_200 = batch_200.to(self.device200, non_blocking=True) # (B*5,12,1,200)
		features_cls200 = self.model200(batch_200) #(B*5,384)

		#40采样点token间注意力，[cls]代表200采样点token=>200采样点中哪段40采样点最“重要”
		attention_200 = self.model200.get_last_selfattention(batch_200) #(B*5,num_heads,6,6)
		nh = attention_200.shape[1] # number of head==6
		attention_200 = attention_200[:, :, 0, 1:].reshape(B*h_200, nh, -1) # [CLS] token 到每个 slice token 的注意力值:(B*5,nh,5)->(B*5,nh,5)
		attention_200 = attention_200.cpu().numpy()



		features_grid200 = features_cls200.reshape(B, h_200, 384).unfold(1,5,5).reshape(B,384,1,5) # (B,384,1,5)
		features_grid200 = features_grid200.to(self.device1k, non_blocking=True)
		features_cls1k = self.model1k.forward(features_grid200).detach().cpu()

		attention_1k = self.model1k.get_last_selfattention(features_grid200) #(B,nh,6,6)
		nh = attention_1k.shape[1] # number of head
		attention_1k = attention_1k[0, :, 0, 1:].reshape(B,nh, -1) #(B,nh,5)
		attention_1k = attention_1k.cpu().numpy()

		return tensorbatch2im(batch_200), attention_200, attention_1k #(B*5,200,12), (B*5,nh,5),(B,nh,5)



	def prepare_ser_tensor(self, ser: torch.Tensor, slice_len=200):
		"""
		裁剪 ECG 序列的最后一个维度（长度 L），使其能被 slice_len 整除，保留中间部分。

		Args:
			ser (torch.Tensor): ECG序列，形状 [B x C x 1 x H']
			slice_len (int): 要整除的单位长度，默认是200

		Returns:
			ser_new (torch.Tensor): 裁剪后的ECG序列
			num_slices (int): 裁剪后的序列中包含多少个slice_len
		"""

		assert ser.dim() == 4 and ser.shape[2] == 1, "输入必须是 [B, C, 1, L] 的形状哦～"

		l = ser.shape[3]
		dummy_dim = ser.shape[2]
		new_l = l - (l % slice_len)
		start = (l - new_l) // 2
		end = start + new_l

		ser_new = ser[:, :, :, start:end]
		num_slices = new_l // slice_len
		

		return ser_new, dummy_dim, num_slices