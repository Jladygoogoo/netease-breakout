import os
import os
import sys
import json
import pickle
import numpy as np 
import random
random.seed(21)
from collections import Counter

from gensim.models import Doc2Vec, Word2Vec

import torch
import torch.utils.data as data

from config import Config

# 将codes包（__init__.py将文件夹变成包）载入import路径列表中
sys.path.append("/Users/inkding/Desktop/netease2/codes")
from utils import get_mfcc, get_d2v_vector, get_w2v_vector
from connect_db import MyConn

class MyDataste(data.Dataset):
	'''
	将对应的音频特征、歌词特征、评论中的语义特征打包到一起
	+ music feature(mfcc): (20, 1292)
	+ lyrics feature(doc2vec vector): (300,)
	+ reviews feature(feature words): K * (300,)
	'''
	def __init__(self, config, train=True):
		'''
		+ breakouts: 爆发点的数组集合，每个爆发点以字典的形式记录（track_id, date, beta, reviews_num, feature_words）
		+ ids: 爆发点的索引，用于__getitem__方法
		+ conn: 数据路连接，用于获取数据路径
		+ w2v/d2v: 预训练Word2Vec和Doc2Vec模型
		'''
		self.config = config
		self.conn = MyConn()

		if train:
			self.breakouts = open(config.breakouts_id_train).read().splitlines()
			self.no_breakouts = open(config.no_breakouts_id_train).read().splitlines()
		else:
			self.breakouts = open(config.breakouts_id_test).read().splitlines()
			self.no_breakouts = open(config.no_breakouts_id_test).read().splitlines()

		self.ids = list(range(len(self.breakouts)+len(self.no_breakouts)))
		print(len(self.breakouts)+len(self.no_breakouts))

		self.d2v_model = Doc2Vec.load(config.d2v_path)
		self.w2v_model = Word2Vec.load(config.w2v_path)
		

	def __getitem__(self, index):
		'''
		必须自定义，使用index获取成对数据
		'''
		if index<len(self.breakouts):
			label = 1
			id_ = self.breakouts[index]
			track_id, beta = self.conn.query(targets=["track_id", "beta"], table="breakouts",
												conditions={"id":id_})[0]
			feature_words = self.conn.query(targets=["feature_words"], table="breakouts_feature_words_1",
												conditions={"breakout_id":id_})[0][0].split()
		else:
			label = 0
			index = index - len(self.breakouts)
			id_ = self.no_breakouts[index]
			track_id = id_.split('-')[0]
			beta = 1
			feature_words = self.conn.query(targets=["feature_words"], table="no_breakouts_feature_words_1",
												conditions={"id":id_})[0][0].split()

		rawmusic_path, lyrics_path = self.conn.query(
			targets=["rawmusic_path", "lyrics_path"], conditions={"track_id": track_id})[0]

		mfcc = torch.Tensor(get_mfcc(rawmusic_path))
		lyrics_vec = torch.Tensor(get_d2v_vector(lyrics_path, self.d2v_model))

		return label, beta, mfcc, lyrics_vec, feature_words


	def __len__(self):
		'''
		必须自定义，返回总样本数量
		'''
		return len(self.ids)



def get_loader(config, train=True):
	'''
	为MyDataset返回torch.utils.data.DataLoader对象
	'''
	my_dataset = MyDataste(config, train=train)
	data_loader = data.DataLoader(dataset=my_dataset,
								batch_size=config.batch_size,
								shuffle=True)

	return my_dataset, data_loader





if __name__ == '__main__':
	config = Config()
	dataset, data_loader = get_loader(config)
	

	for i, (label, beta, mfcc, lyrics_vec, feature_words) in enumerate(data_loader):
		print(lyrics_vec[0]-lyrics_vec[1])
		break

	# conn = MyConn()
	# tracks = set()
	# with open(json_path) as f:
	# 	content = json.load(f)
	# for item in content:
	# 	tid = item["track_id"]
	# 	if tid not in tracks:
	# 		tracks.add(tid)
	# 		rawmusic_path, lyrics_path = conn.query(targets=["rawmusic_path","lyrics_path"],
	# 												conditions={"track_id":tid})[0]
	# 		if (not os.path.exists(rawmusic_path)) or (not os.path.exists("/Volumes/nmusic/NetEase2020/data" + lyrics_path)):
	# 			print(tid)

