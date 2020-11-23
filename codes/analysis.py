import os
import json
import pandas as pd
import numpy as np
import pickle
from collections import Counter

import librosa
import matplotlib.pyplot as plt 
import seaborn as sns 
from pyecharts.charts import Bar
from pyecharts import options as opts

from connect_db import MyConn
from utils import get_every_day


def draw_hist(data, **kwargs):
	sns.displot(data, **kwargs)
	plt.show()


def draw_bar(data, render_path):
	print(data)
	bar = (
		Bar(init_opts={'width':'1200px','height':'2000px'})
		.add_xaxis(list(data.keys()))
		.add_yaxis(
			"爆发", 
			list(zip(*list(data.values())))[0],
			itemstyle_opts=opts.ItemStyleOpts(color='#499c9f')
		)
		.add_yaxis(
			"未爆发", 
			list(zip(*list(data.values())))[1],
			itemstyle_opts=opts.ItemStyleOpts(color='#FFA421')
		)
		.reversal_axis()
		.set_series_opts(label_opts=opts.LabelOpts(position="right"))
		.set_global_opts(title_opts=opts.TitleOpts(title="自带标签统计"))
	)
	bar.render(render_path)



def basic_analysis(tracks_set):
	'''
	对指定的歌曲集进行基本分析：评论数、时间跨度...
	'''
	conn = MyConn()
	# 数据准备
	data = []
	targets = ["track_id", "tags", "reviews_num", "first_review", "last_review"]
	for tid in tracks_set:
		res = conn.query(targets=targets, conditions={"track_id": int(tid)})
		data.append(res[0])
	
	df = pd.DataFrame(data, columns=targets)
	# df.to_csv("../results/main_tagged_tracks/basic_info.csv", encoding="utf_8_sig", index=False)

	# hist_view(df["reviews_num"].values, log_scale=True, color="tab:orange")
	durations = list(df.apply(lambda d: len(get_every_day(d["first_review"], d["last_review"], str_source=False)), axis=1).array)
	draw_hist(durations)

	# tag

def mp3_analysis(tracks_set):
	'''
	对指定的歌曲集的音频时长进行分析。
	'''
	# 数据准备
	data_path = "../data/main_tagged_tracks/music_preload_data.pkl"
	with open(data_path,'rb') as f:
		data = pickle.load(f)

	durations = list(map(lambda x: librosa.get_duration(x[0],x[1]), data.values()))
	print("max duration:", np.max(durations))
	print("min duration:", np.min(durations))
	hist_view(durations)



def in_tags_analysis(breakouts_set, no_breakouts_set):
	'''
	对指定的歌曲集的内置tags情况进行分析。
	'''
	tags = open("../data/metadata/自带tags.txt").read().splitlines()
	breakouts_tags_d = {}
	no_breakouts_tags_d = {}
	for t in tags:
		breakouts_tags_d[t] = []
		no_breakouts_tags_d[t] = []

	conn = MyConn()
	for tid in breakouts_set:
		res = conn.query(targets=["tags"], conditions={"track_id":tid})[0]
		for t in res[0].split():
			breakouts_tags_d[t].append(tid)
	for tid in no_breakouts_set:
		res = conn.query(targets=["tags"], conditions={"track_id":tid})[0]
		for t in res[0].split():
			no_breakouts_tags_d[t].append(tid)

	tags_count = []
	for k in breakouts_tags_d:
		tags_count.append((k, (float(format(len(breakouts_tags_d[k])/1748*100,'.2f')), 
								float(format(len(no_breakouts_tags_d[k])/10,'.2f')))))

	tags_count = sorted(tags_count, key=lambda x:x[1][0], reverse=False)
	draw_bar(dict(tags_count), "../data/main_tagged_tracks/tags_count.html")


def breakouts(json_path="../data/breakouts-u2.json"):
	'''
	基于json文件进行分析。
	'''
	# with open(json_path) as f:
	# 	content = json.load(f)
	df = pd.read_json(json_path)
	# tracks = list(df["track_id"].unique())
	# print(df.shape)
	# print(len(tracks))
	# basic_analysis(tracks)
	values = df["reviews_num"].values
	print(np.median(values))
	draw_hist(list(filter(lambda x:x<=2000, values)), color="pink")
	# draw_hist(values)


def pos_neg_words():
	'''
	考察pairwise训练中的正负样本情况。
	'''
	path = "MyModel/models/pos_neg_words-3.pkl"
	with open(path, "rb") as f:
		content = pickle.load(f)
	for batch in content.values():
		neg_words = []
		for item in batch:
			neg_words.extend(item[1])
		print(Counter(neg_words).most_common(10))



if __name__ == '__main__':
	# basic_analysis(tracks_set)
	# mp3_analysis(tracks_set)
	# in_tags_analysis(tracks_set, tracks_set2)
	# breakouts()
	with open("MyModel/models/3/losses.pkl", "rb") as f:
		losses = pickle.load(f)
	x = range(0, len(losses)*10, 10)
	plt.plot(x, list(losses.values()))
	plt.xlabel("batch")
	plt.ylabel("loss")
	plt.title("my_loss2")
	plt.show()
	# pos_neg_words()

