import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class MusicFeatureExtractor(nn.Module):
	'''
	利用CNN网络从mfcc_frames中提取音频特征
	'''
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=48, 
					kernel_size=3, stride=1, padding=1), # 不改变W和H，C变为48
			nn.BatchNorm2d(num_features=48), # num_features为当前数据的通道数
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4), stride=(2,4))
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=48, out_channels=96, 
					kernel_size=3, stride=1, padding=1), # 不改变W和H，C变为96
			nn.BatchNorm2d(num_features=96), # num_features为当前数据的通道数
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4), stride=(2,4))
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=96, out_channels=192, 
					kernel_size=3, stride=1, padding=1), # 不改变W和H，C变为192
			nn.BatchNorm2d(num_features=192), # num_features为当前数据的通道数
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,4), stride=(2,4))
		)
		self.fc = nn.Linear(in_features=7680, out_features=config.cnn_out_size)

	def forward(self, x):
		'''
		params: x: mfcc features, shape = (batch_size, 20, 1292)
		return: output: ()
		'''
		# CNN部分
		# nn.Conv2d: input/output size: (N, C, H, W)
		x = x.unsqueeze(1) # (batch_size, 1, 20, 1292)
		# print(x.shape)
		x = self.conv1(x) # (batch_size, 48, 10, 323)
		# print(x.shape)
		x = self.conv2(x) # (batch_size, 96, 5, 80)
		# print(x.shape)
		x = self.conv3(x) # (batch_size, 192, 2, 20)
		# print(x.shape)
		x = x.view(x.size(0), -1) # (batch_size, 7680)
		# print(x.shape)
		output = self.fc(x)
		return output


class IntrinsicFeatureEmbed(nn.Module):
	'''
	将固有特征组合并映射到嵌入空间
	'''
	def __init__(self, config):
		super().__init__()
		self.transform = nn.Sequential(
			nn.LayerNorm(1324),
			nn.Linear(in_features=1324, out_features=config.embedding_size)
		)

	def forward(self, x):
		return self.transform(x)



class SemanticFeatureEmbed(nn.Module):
	'''
	将语义特征映射到嵌入空间
	'''
	def __init__(self, config):
		super().__init__()
		self.transform = nn.Linear(in_features=300, out_features=config.embedding_size)

	def forward(self, x):
		return self.transform(x)

