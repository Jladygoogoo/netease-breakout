import os
import sys
import pickle
import time
import numpy as np
import pandas as pd 
import logging
import traceback

from gensim.models import Doc2Vec

import torch

from connect_db import MyConn
from utils import get_mfcc, get_d2v_vector, get_tags_vector, roc_auc_score, time_spent
from MyModel.model import MusicFeatureExtractor, IntrinsicFeatureEmbed
from MyModel.config import Config


def concatenate_features(mfcc, lyrics_vec, tags_vec):
	feature_vec = np.concatenate((mfcc.ravel(), lyrics_vec, tags_vec))
	return feature_vec

# 构建.pkl文件的路径字典
def get_tid_2_mp3path_d(dir_):
	tid_2_mp3path_d = {}
	# mp3_r_path = "../data/main_tagged_tracks/music_preload_data"
	for root, dirs, files in os.walk(dir_):
		for file in files:
			if "DS" in file: continue
			tid_2_mp3path_d[file[:-4]] = os.path.join(root, file)
	return tid_2_mp3path_d


def get_X_y(tracks_2_labels, conn, d2v_model):
	# 构建.pkl文件的路径字典
	tid_2_mp3path_d = get_tid_2_mp3path_d()

	X = []
	y = []
	flag = 1
	for tid, label in tracks_2_labels.items():
		try:
			# 获取音频特征向量
			mfcc = get_mfcc(tid_2_mp3path_d[tid])
			# 获取歌词特征向量
			lyrics_path, tags = conn.query(targets=["lyrics_path","tags"],
											conditions={"track_id":tid})[0]
			lyrics_vec = get_d2v_vector("/Volumes/nmusic/NetEase2020/data/"+lyrics_path, d2v_model)
			# 获取标签特征向量
			tags_vec = get_tags_vector(tags.split())

			feature_vec = concatenate_features(mfcc, lyrics_vec, tags_vec)
			# feature_vec = np.concatenate((mfcc.ravel(), lyrics_vec))
			X.append(feature_vec)
			y.append(label)

			print(flag, tid)
			flag += 1
		except KeyboardInterrupt:
			print("interrupted by user.")
			sys.exit(1)
		except:
			print("ERROR", tid)
			print(traceback.format_exc())
			# continue
			# break
	return X, y


def get_X_y_embed(tracks_2_labels, conn, d2v_model, music_feature_extractor, intrinsic_feature_embed):
	# 构建.pkl文件的路径字典
	dir1 = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
	dir2 = "/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic"
	tid_2_mp3path_d = get_tid_2_mp3path_d(dir1)
	tid_2_mp3path_d2 = get_tid_2_mp3path_d(dir2)
	for tid in tid_2_mp3path_d2:
		if tid not in tid_2_mp3path_d:
			tid_2_mp3path_d[tid] = tid_2_mp3path_d2[tid]
	# print(len(tid_2_mp3path_d))

	X = []
	y = []
	flag = 1
	for tid, label in tracks_2_labels.items():
		tid = str(tid)
		try:
			# 获取音频特征向量
			mfcc = torch.tensor(get_mfcc(tid_2_mp3path_d[tid])).unsqueeze(0)
			# 获取歌词特征向量
			lyrics_path = conn.query(targets=["lyrics_path"],
											conditions={"track_id":tid})[0][0]
			lyrics_vec = torch.tensor(get_d2v_vector(
				"/Volumes/nmusic/NetEase2020/data/"+lyrics_path, d2v_model))

			h1 = music_feature_extractor(mfcc).squeeze()
			# print(h1.shape)
			h2 = torch.cat((h1, lyrics_vec))
			# print(h2.shape)
			feature_vec = intrinsic_feature_embed(h2)

			X.append(feature_vec)
			y.append(label)

			print(flag, tid)
			flag += 1
		except KeyboardInterrupt:
			print("interrupted by user.")
			sys.exit(1)
		except:
			print("ERROR", tid)
			print(traceback.format_exc())
			# continue
			# break
	return X, y

def build_dataset():
	filter_tracks = ["442314990", "5263408", "29418974", "742265"]
	ts1 = open("../data/main_tagged_tracks/tracks.txt").read().splitlines()
	ts2 = open("../data/main_tagged_tracks/no_breakouts_tracks.txt").read().splitlines()

	# 是否爆发
	tracks_set = [(tid,1) for tid in ts1 if tid not in filter_tracks]
	tracks_set += [(tid,0) for tid in ts2 if tid not in filter_tracks]

	# 之后从mysql数据库中获取lyrics路径和tag
	conn = MyConn()
	d2v_model = Doc2Vec.load("../models/d2v/d2v_a2.mod")

	X, y = get_X_y(dict(tracks_set), conn, d2v_model)

	with open("../data/main_tagged_tracks/dataset_violent_a2.pkl", 'wb') as f:
		pickle.dump([X,y], f)


def build_dataset_multiclass():
	filter_tracks = ["442314990", "5263408", "29418974", "742265"]
	with open("../data/main_tagged_tracks/labels_dict.pkl",'rb') as f:
		content = pickle.load(f)
	labels = list(content.keys())
	# 将字典反转
	tid_2_label_d = {}
	for k,v in content.items():
		for tid in v:
			tid = str(tid)
			if tid not in tid_2_label_d:
				tid_2_label_d[tid] = labels.index(k)+1

	no_breakouts = open("../data/main_tagged_tracks/no_breakouts_tracks.txt").read().splitlines()
	for tid in no_breakouts:
		tid_2_label_d[tid] = 0 # 0表示没有爆发

	for tid in filter_tracks:
		del tid_2_label_d[tid]


	# 之后从mysql数据库中获取lyrics路径和tag
	conn = MyConn()
	d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

	X, y = get_X_y(tid_2_label_d, conn, d2v_model)

	with open("../data/main_tagged_tracks/dataset_violent_multiclass.pkl", 'wb') as f:
		pickle.dump([X,y], f)



def build_dataset_multilabels():
	filter_tracks = ["442314990", "5263408", "29418974", "742265"]
	with open("../data/main_tagged_tracks/labels_dict.pkl",'rb') as f:
		content = pickle.load(f)
	labels = list(content.keys())
	# 将字典反转
	tid_2_label_d = {}
	for k,v in content.items():
		for tid in v:
			tid = str(tid)
			if tid in tid_2_label_d:
				tid_2_label_d[tid].append(labels.index(k)+1)
			else:
				tid_2_label_d[tid] = [labels.index(k)+1]

	no_breakouts = open("../data/main_tagged_tracks/no_breakouts_tracks.txt").read().splitlines()
	for tid in no_breakouts:
		tid_2_label_d[tid] = [0] # 0表示没有爆发

	for tid in filter_tracks:
		del tid_2_label_d[tid]


	# 之后从mysql数据库中获取lyrics路径和tag
	conn = MyConn()
	d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

	multi_labels = MultiLabelBinarizer().fit_transform(list(tid_2_label_d.values()))
	tid_2_label_d = dict(zip(list(tid_2_label_d.keys()), multi_labels))
	

	X, y = get_X_y(tid_2_label_d, conn, d2v_model)

	with open("../data/main_tagged_tracks/dataset_violent_multilabels.pkl", 'wb') as f:
		pickle.dump([X,y], f)



def build_dataset_less():
	filter_tracks = ["442314990", "5263408", "29418974", "742265"]
	ts1 = open("../data/main_tagged_tracks/tracks.txt").read().splitlines()[:1000]
	ts2 = open("../data/main_tagged_tracks/no_breakouts_tracks.txt").read().splitlines()

	# 是否爆发
	tracks_set = [(tid,1) for tid in ts1 if tid not in filter_tracks]
	tracks_set += [(tid,0) for tid in ts2 if tid not in filter_tracks]

	# 之后从mysql数据库中获取lyrics路径和tag
	conn = MyConn()
	d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

	X, y = get_X_y(dict(tracks_set), conn, d2v_model)

	with open("../data/main_tagged_tracks/dataset_violent_less.pkl", 'wb') as f:
		pickle.dump([X,y], f)



def build_dataset_embed(w_path):
	# ts1 = open("../data/main_tagged_tracks/tracks.txt").read().splitlines()[:1000]
	ts1 = list(pd.read_json("../data/breakouts-u2.json")["track_id"].unique())[:1000]
	ts2 = open("../data/no_breakouts_tracks.txt").read().splitlines()[:1000]
	print(len(ts1), len(ts2))

	# 是否爆发
	tracks_set = [(tid,1) for tid in ts1]
	tracks_set += [(tid,0) for tid in ts2]

	# 加载模型
	conn = MyConn()
	d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

	config = Config()
	mf_path = "MyModel/models/3/mf_extractor-e3.pkl"
	if_path = "MyModel/models/3/if_embed-e3.pkl"
	music_feature_extractor = MusicFeatureExtractor(config)
	music_feature_extractor.load_state_dict(torch.load(mf_path))
	intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
	intrinsic_feature_embed.load_state_dict(torch.load(if_path))
	music_feature_extractor.eval()
	intrinsic_feature_embed.eval()

	X, y = get_X_y_embed(dict(tracks_set), conn, d2v_model, music_feature_extractor, intrinsic_feature_embed)

	with open(w_path, 'wb') as f:
		pickle.dump([X,y], f)


if __name__ == '__main__':
	w_path = "../data/mymodel_data/dataset_embed-3-1000.pkl"
	build_dataset_embed(w_path)


