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
	def __init__(self, config):
		'''
		+ breakouts: 爆发点的数组集合，每个爆发点以字典的形式记录（track_id, date, beta, reviews_num, feature_words）
		+ ids: 爆发点的索引，用于__getitem__方法
		+ conn: 数据路连接，用于获取数据路径
		+ w2v/d2v: 预训练Word2Vec和Doc2Vec模型
		'''
		self.config = config
		self.ids = list(range(config.breakout_size + config.no_breakout_size))
		self.conn = MyConn()
		self.breakouts = random.sample([r[0] for r in self.conn.query(targets=["breakout_id"], conditions={"have_words": 1, "have_rawmusic":1}, table="breakouts")], 
										config.breakout_size)
		self.no_breakouts = random.sample(open(config.no_breakouts_file).read().splitlines(),
										config.no_breakout_size)
		self.d2v_model = Doc2Vec.load(config.d2v_path)
		self.w2v_model = Word2Vec.load(config.w2v_path)
		

	def __getitem__(self, index):
		'''
		必须自定义，使用index获取成对数据
		'''
		if index<self.config.breakout_size:
			label = 1
			breakout_id = self.breakouts[index]
			track_id, beta = self.conn.query(targets=["track_id", "beta"], table="breakouts",
												conditions={"breakout_id":breakout_id})[0]
			feature_words = self.conn.query(targets=["feature_words"], table="breakouts_feature_words_1",
												conditions={"breakout_id":breakout_id})[0][0].split()
		else:
			label = 0
			index = index - self.config.breakout_size
			track_id = self.no_breakouts[index]
			beta = 1
			# feature_words = self.conn.query(targets=["feature_words"], table="no_breakout_feature_words-1",
			# 									conditions={"track_id":track_id})[0][0].split()
			feature_words = ""

		rawmusic_path, lyrics_path = self.conn.query(
			targets=["rawmusic_path", "lyrics_path"], conditions={"track_id": track_id})[0]
		lyrics_path = "/Volumes/nmusic/NetEase2020/data" + lyrics_path

		mfcc = torch.Tensor(get_mfcc(rawmusic_path))
		lyrics_vec = torch.Tensor(get_d2v_vector(lyrics_path, self.d2v_model))

		return label, beta, mfcc, lyrics_vec, feature_words


	def __len__(self):
		'''
		必须自定义，返回总样本数量
		'''
		return len(self.ids)



def get_loader(config):
	'''
	为MyDataset返回torch.utils.data.DataLoader对象
	'''
	my_dataset = MyDataste(config)
	data_loader = data.DataLoader(dataset=my_dataset,
								batch_size=config.batch_size) #, shuffle=True

	return my_dataset, data_loader





if __name__ == '__main__':
	json_path = "/Users/inkding/Desktop/netease2/data/breakouts-u1.json"
	d2v_path = "/Users/inkding/Desktop/netease2/models/d2v/d2v_a1.mod"
	w2v_path = "/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod"
	batch_size = 32
	dataset, data_loader = get_loader(json_path=json_path,
										d2v_path=d2v_path, 
										w2v_path=w2v_path,
										batch_size=batch_size)
	


	# for i, (beta, mfcc, lyrics_vec, feature_words) in enumerate(data_loader):
	# 	pass

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

