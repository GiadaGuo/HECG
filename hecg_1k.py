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

from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Local Dependencies
import ecg_vit1 as vits
import ecg_vit2 as vits1k
# from HECG_heatmap_utils import *
from hecg_model_utils import get_vit200, get_vit1k,eval_transforms,tensorbatch2im


class HECG_1K(torch.nn.Module):
	"""
	HECG Model (ViT-1K) for encoding ECG series (with 200 slice tokens), with 200 slice tokens
	encoded via ViT-200 using 40 slice tokens.
	"""
	def __init__(self,
		#这里本地文件太大传不上去只能用云盘的，等加载再改吧
		model200_path: str = '../Checkpoints/vit200_small_dino.pth',
		model1k_path: str = '../Checkpoints/vit1k_xs_dino.pth', 
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
	
	def forward(self, x):
		"""
		Forward pass of HECG (given an series tensor x), outputting the [CLS] token from ViT-1k.
		1. x is center-cropped(中间剪裁) such that the W / H is divisible by the slice token size in ViT-1k (e.g. - 200).
		概括：中间裁剪 x → 确保尺寸能被 200 整除
		这个功能暂时不用，默认输入都是5000采样点

		2. x then gets unfolded into a "batch" of [200] series.
		概括：切割 x → 变成多个 采样点200 的子序列

		3. A pretrained ViT-200 model extracts the CLS token from each [200] series in the batch.
		概括：ViT-200 处理子列 → 获取 [CLS]_200 特征

		4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_200" and height "h_200".)
		概括：重新组织[CLS]_256特征 → 形成 2D 特征网格（增加虚拟维度w_200=1）


		5. This feature grid is then used as the input to ViT-1k, outputting [CLS]_1k.
		概括：ViT-1k 整合特征 → 得到最终 [CLS]_1k
		
		Args:
			- x (torch.Tensor): [1 x C x 1 x H'] image tensor.
			输入：形状是 [1 × C × 1 × H']，即 1 张 12导联采样点H的心电图样本，经过裁剪后的大小是 1 × H'
		
		Return:
			- features_cls1k (torch.Tensor): [1 x 192] cls token (d_1k = 192 by default).
			输出：形状是 [1 × 192]，表示整个 x 的全局特征，192 维的1个向量
		"""
		batch_200, w_200, h_200 = self.prepare_ser_tensor(x)                    # 1. batch_200：[1 x 12 x 1 x H]=>e.g.[1 x 12 x 1 x 1000];h_200通常是5

		batch_200 = batch_200.unfold(3, 200, 200)           # 2. [1, 12, 1, 5, 200]
		batch_200 = rearrange(batch_200, 'b c w n h -> (b n) c w h')    # 2. [B x 12 x 1 x 200], where B = (1*w_200*h_200) 表示batchsize，成为batch_200.shape[0]

		# 所有的B个slices被分成B/5个minibatches（每个里面有5个slices）
		# 按minibatch进model(VIT_small)训练，得到的CLS添加进features_cls200列表
		# input size: [5 x 12 x 1 x 200] （进模型后200采样点会被打散成5个40采样点的子列）
		# output size: [5 x 384]（每个minibatch里有5个slices，每个slices的大小是200，最终都生成5个1*384的向量）
		features_cls200 = []

		# #B如果很大用minibatch
		# for mini_bs in range(0, batch_200.shape[0], 5):                       # 3. B may be too large for ViT-200. We further take minibatches of 5.(每隔5取batchsize里的序号)
		# 	minibatch_200 = batch_200[mini_bs:mini_bs+5].to(self.device200, non_blocking=True)
		# 	#self.model200接受的输入维度： [B x 12 x 1 x 200]
		# 	features_cls200.append(self.model200(minibatch_200).detach().cpu()) # 3. Extracting ViT-200 features from [256 x 3 x 256 x 256] image batches.
		# 	#+3.提取后的特征被移动到 CPU 上，并从计算图中分离（detach）
		#
		# #torch.vstack:将 features_cls200 列表中的所有张量沿着第一个维度（即垂直方向）堆叠起来,把minibatches还原成B
		# #把B/256个[5*384]堆叠——[B*384],B是最大的输入图像可以分成1*200的数量
		# features_cls200 = torch.vstack(features_cls200)                         # 3. [B x 384], where 384 == dim of ViT-200 [ClS] token.

		#B不大不用minibatch：B=5左右
		batch_200=batch_200.to(self.device200, non_blocking=True)
		features_cls200 = self.model200(batch_200).detach().cpu()

		#维度变换还原成最开始的2D输入形式 (1,C,W,H)——(1,384, w_200,h_200)，便于下一层输入
		#过程：[B*384]→(w_200, h_200, 384)→交换0和1→(h_200, w_200, 384)→交换0和2→(384,w_200,h_200)→unsqueeze(dim=0)在第0维增加1个维度→(1,384,w_200,h_200)
		features_cls200 = features_cls200.reshape(w_200, h_200, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0) 

		#【进入下一层】
		features_cls200 = features_cls200.to(self.device1k, non_blocking=True)  # 4. [1 x 384 x w_200 x h_200]；non_blocking=True 可以让数据传输操作在后台进行，而不会阻塞当前线程
		features_cls1k = self.model1k.forward(features_cls200)                  # 5. [1 x 192], where 192 == dim of ViT-1k [ClS] token.
		return features_cls1k
	
	
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


	def _get_signal_attention_scores(self, signal, scale=1):
		r"""
		Forward pass in hierarchical model with attention scores saved.
		
		Args:
		- signal (PIL.Image):       ECG series with 1000 sampling points
		- model200 (torch.nn):      200-Level ViT
		- model1k (torch.nn):       1000-Level ViT
		- scale (int):              How much to scale the output image by (e.g. - scale=4 will resize signal to be 250)
		
		Returns:
		- np.array: [5, 1, 200/scale, 12] np.array sequence of image slices from the 1000 sampling points series.
		- attention_200 (torch.Tensor): [5, 1, 200/scale, 12] torch.Tensor sequence of attention maps for 200-sized slices.
		- attention_1k (torch.Tensor): [1, 1, 1000/scale, 12] torch.Tensor sequence of attention maps for 1k-sized series.
		"""

		x = eval_transforms(signal).unsqueeze(dim=0)

		batch_200, w_200, h_200 = self.prepare_ser_tensor(x)
		batch_200 = batch_200.unfold(3, 200, 200)
		batch_200 = rearrange(batch_200, 'b c w n h -> (b n) c w h')
		batch_200 = batch_200.to(self.device200, non_blocking=True)
		features_cls200 = self.model200(batch_200)

		attention_200 = self.model200.get_last_selfattention(batch_200) #(B,num_heads,特征数,特征数)=>(5,num_heads,6,6)
		nh = attention_200.shape[1] # number of head
		attention_200 = attention_200[:, :, 0, 1:].reshape(5, nh, -1) # [CLS] token 到每个 patch token 的注意力值:(5,nh,5)->(5,nh,5)
		attention_200 = attention_200.reshape(w_200*h_200, nh, 1, 5) #(5,nh,5)->(5,nh,1,5)
		attention_200 = nn.functional.interpolate(attention_200, scale_factor=int(40/scale), mode="nearest").cpu().numpy()

		features_grid200 = features_cls200.reshape(w_200, h_200, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
		features_grid200 = features_grid200.to(self.device1k, non_blocking=True)
		features_cls1k = self.model1k.forward(features_grid200).detach().cpu()

		attention_1k = self.model1k.get_last_selfattention(features_grid200) #(1,nh,6,6)
		nh = attention_1k.shape[1] # number of head
		attention_1k = attention_1k[0, :, 0, 1:].reshape(nh, -1) #(nh,5)
		attention_1k = attention_1k.reshape(nh, w_200, h_200) #(nh,1,5)
		attention_1k = nn.functional.interpolate(attention_1k.unsqueeze(0), scale_factor=int(200/scale), mode="nearest")[0].cpu().numpy()

		if scale != 1:
			batch_200 = nn.functional.interpolate(batch_200, scale_factor=(1/scale), mode="nearest")

		return tensorbatch2im(batch_200), attention_200, attention_1k


	# def get_region_attention_heatmaps(self, x, offset=128, scale=4, alpha=0.5, cmap = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet), threshold=None):
	# 	r"""
	# 	Creates hierarchical heatmaps (Raw H&E + ViT-200 + ViT-1k + Blended Heatmaps saved individually).
	#
	# 	Args:
	# 	- region (PIL.Image):       4096 x 4096 Image
	# 	- model200 (torch.nn):      256-Level ViT
	# 	- model1k (torch.nn):       4096-Level ViT
	# 	- output_dir (str):         Save directory / subdirectory
	# 	- fname (str):              Naming structure of files
	# 	- offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending
	# 	- scale (int):              How much to scale the output image by
	# 	- alpha (float):            Image blending factor for cv2.addWeighted
	# 	- cmap (matplotlib.pyplot): Colormap for creating heatmaps
	#
	# 	Returns:
	# 	- None
	# 	"""
	# 	region = Image.fromarray(tensorbatch2im(x)[0])
	# 	w, h = region.size
	#
	# 	region2 = add_margin(region.crop((128,128,w,h)),
	# 					 top=0, left=0, bottom=128, right=128, color=(255,255,255))
	# 	region3 = add_margin(region.crop((128*2,128*2,w,h)),
	# 					 top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
	# 	region4 = add_margin(region.crop((128*3,128*3,w,h)),
	# 					 top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
	#
	# 	b256_1, a256_1, a1k_1 = self._get_region_attention_scores(region, scale)
	# 	b256_2, a256_2, a1k_2 = self._get_region_attention_scores(region, scale)
	# 	b256_3, a256_3, a1k_3 = self._get_region_attention_scores(region, scale)
	# 	b256_4, a256_4, a1k_4 = self._get_region_attention_scores(region, scale)
	# 	offset_2 = (offset*1)//scale
	# 	offset_3 = (offset*2)//scale
	# 	offset_4 = (offset*3)//scale
	# 	w_s, h_s = w//scale, h//scale
	# 	w_200, h_200 = w//256, h//256
	# 	save_region = np.array(region.resize((w_s, h_s)))
	#
	# 	if threshold != None:
	# 		for i in range(6):
	# 			score256_1 = concat_scores256(a256_1[:,i,:,:], w_200, h_200, size=(w_s//w_200,h_s//h_200))
	# 			score256_2 = concat_scores256(a256_2[:,i,:,:], w_200, h_200, size=(w_s//w_200,h_s//h_200))
	# 			new_score256_2 = np.zeros_like(score256_2)
	# 			new_score256_2[offset_2:w_s, offset_2:h_s] = score256_2[:(w_s-offset_2), :(h_s-offset_2)]
	# 			overlay256 = np.ones_like(score256_2)*100
	# 			overlay256[offset_2:w_s, offset_2:h_s] += 100
	# 			score256 = (score256_1+new_score256_2)/overlay256
	#
	# 			mask256 = score256.copy()
	# 			mask256[mask256 < threshold] = 0
	# 			mask256[mask256 > threshold] = 0.95
	#
	# 			color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
	# 			region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
	# 			region256_hm[mask256==0] = 0
	# 			img_inverse = save_region.copy()
	# 			img_inverse[mask256 == 0.95] = 0
	# 			Image.fromarray(region256_hm+img_inverse).save(os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))
	#
	# 	if False:
	# 		for j in range(6):
	# 			score1k_1 = concat_scores1k(a1k_1[j], size=(h_s,w_s))
	# 			score1k = score1k_1 / 100
	# 			color_block1k = (cmap(score1k)*255)[:,:,:3].astype(np.uint8)
	# 			region1k_hm = cv2.addWeighted(color_block1k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
	# 			Image.fromarray(region1k_hm).save(os.path.join(output_dir, '%s_1k[%s].png' % (fname, j)))
	#
	# 	hm1k, hm256, hm1k_256 = [], [], []
	# 	for j in range(6):
	# 		score1k_1 = concat_scores1k(a1k_1[j], size=(h_s,w_s))
	# 		score1k_2 = concat_scores1k(a1k_2[j], size=(h_s,w_s))
	# 		score1k_3 = concat_scores1k(a1k_3[j], size=(h_s,w_s))
	# 		score1k_4 = concat_scores1k(a1k_4[j], size=(h_s,w_s))
	# 		new_score1k_2 = np.zeros_like(score1k_2)
	# 		new_score1k_2[offset_2:h_s, offset_2:w_s] = score1k_2[:(h_s-offset_2), :(w_s-offset_2)]
	# 		new_score1k_3 = np.zeros_like(score1k_3)
	# 		new_score1k_3[offset_3:h_s, offset_3:w_s] = score1k_3[:(h_s-offset_3), :(w_s-offset_3)]
	# 		new_score1k_4 = np.zeros_like(score1k_4)
	# 		new_score1k_4[offset_4:h_s, offset_4:w_s] = score1k_4[:(h_s-offset_4), :(w_s-offset_4)]
	#
	# 		overlay1k = np.ones_like(score1k_2)*100
	# 		overlay1k[offset_2:h_s, offset_2:w_s] += 100
	# 		overlay1k[offset_3:h_s, offset_3:w_s] += 100
	# 		overlay1k[offset_4:h_s, offset_4:w_s] += 100
	# 		score1k = (score1k_1+new_score1k_2+new_score1k_3+new_score1k_4)/overlay1k
	#
	# 		color_block1k = (cmap(score1k)*255)[:,:,:3].astype(np.uint8)
	# 		region1k_hm = cv2.addWeighted(color_block1k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
	# 		hm1k.append(Image.fromarray(region1k_hm))
	#
	#
	# 	for i in range(6):
	# 		score256_1 = concat_scores256(a256_1[:,i,:,:], h_200, w_200, size=(256, 256))
	# 		score256_2 = concat_scores256(a256_2[:,i,:,:], h_200, w_200, size=(256, 256))
	# 		new_score256_2 = np.zeros_like(score256_2)
	# 		new_score256_2[offset_2:h_s, offset_2:w_s] = score256_2[:(h_s-offset_2), :(w_s-offset_2)]
	# 		overlay256 = np.ones_like(score256_2)*100
	# 		overlay256[offset_2:h_s, offset_2:w_s] += 100
	# 		score256 = (score256_1+new_score256_2)/overlay256
	# 		color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
	# 		region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
	# 		hm256.append(Image.fromarray(region256_hm))
	#
	# 	for j in range(6):
	# 		score1k_1 = concat_scores1k(a1k_1[j], size=(h_s,w_s))
	# 		score1k_2 = concat_scores1k(a1k_2[j], size=(h_s,w_s))
	# 		score1k_3 = concat_scores1k(a1k_3[j], size=(h_s,w_s))
	# 		score1k_4 = concat_scores1k(a1k_4[j], size=(h_s,w_s))
	#
	# 		new_score1k_2 = np.zeros_like(score1k_2)
	# 		new_score1k_2[offset_2:h_s, offset_2:w_s] = score1k_2[:(h_s-offset_2), :(w_s-offset_2)]
	# 		new_score1k_3 = np.zeros_like(score1k_3)
	# 		new_score1k_3[offset_3:h_s, offset_3:w_s] = score1k_3[:(h_s-offset_3), :(w_s-offset_3)]
	# 		new_score1k_4 = np.zeros_like(score1k_4)
	# 		new_score1k_4[offset_4:h_s, offset_4:w_s] = score1k_4[:(h_s-offset_4), :(w_s-offset_4)]
	#
	# 		overlay1k = np.ones_like(score1k_2)*100
	# 		overlay1k[offset_2:h_s, offset_2:w_s] += 100
	# 		overlay1k[offset_3:h_s, offset_3:w_s] += 100
	# 		overlay1k[offset_4:h_s, offset_4:w_s] += 100
	# 		score1k = (score1k_1+new_score1k_2+new_score1k_3+new_score1k_4)/overlay1k
	#
	# 		for i in range(6):
	# 			score256_1 = concat_scores256(a256_1[:,i,:,:], h_200, w_200, size=(256, 256))
	# 			score256_2 = concat_scores256(a256_2[:,i,:,:], h_200, w_200, size=(256, 256))
	# 			new_score256_2 = np.zeros_like(score256_2)
	# 			new_score256_2[offset_2:h_s, offset_2:w_s] = score256_2[:(h_s-offset_2), :(w_s-offset_2)]
	# 			overlay256 = np.ones_like(score256_2)*100
	# 			overlay256[offset_2:h_s, offset_2:w_s] += 100
	# 			score256 = (score256_1+new_score256_2)/overlay256
	#
	# 			factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
	# 			score = (score1k*overlay1k+score256*overlay256)/(overlay1k+overlay256) #factorize(score256*score1k)
	# 			color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
	# 			region1k_256_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
	# 			hm1k_256.append(Image.fromarray(region1k_256_hm))
	#
	# 	return hm1k, hm256, hm1k_256


	def prepare_ser_tensor(self, ser: torch.Tensor, slice_len=200):
		"""
		裁剪 ECG 序列的最后一个维度（长度 L），使其能被 slice_len 整除，保留中间部分。

		Args:
			ser (torch.Tensor): ECG序列，形状 [1 x C x 1 x H']
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