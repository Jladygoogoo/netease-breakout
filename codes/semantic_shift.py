import os
import sys
import json
import jieba
import pandas as pd
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec

from preprocess import cut, tags_extractor
from utils import to_date
from w2v import words_simi_score
from connect_db import MyConn


def get_reviews_partitions(filepath, w2v_model, thres_count=150, thres_simi=0.2, merge_num=2, topk=5):
	'''
	将歌曲评论按照语义划分。
	params:
		filepath: 评论文件路径
		w2v_model: 词向量模型
		thres_count: 合成一段语料的最少评论数
		thres_simi: 合成一段语料的最小相似度
		merge_num: 合并次数
		topk: 合成时考察的feature_words数目
	return: 
		d_reviews_partitions[dict]: d[(start_date,end_date)] = feature_words
	'''
	stops_sup = open("../resources/rubbish_words.txt").read().splitlines()

	with open(filepath) as f:
		df = pd.read_json(f)
	track_id = filepath[:-5]
	df["date"] = df["time"].map(lambda x:to_date(x)) # 将时间戳转为日期
	df.sort_values(by="date", inplace=True)
	p_reviews = list(dict(df.groupby("date")["content"].sum()).items()) # 合并同一日期下的评论
	reviews_count = list(df.groupby("date")["content"].count().values) # 统计各日期下的评论数
	dates = list(df.groupby("date")["content"].count().index) # 得到有效日期
	tmp_count, flag = 0, 0
	tmp_start_flag, tmp_end_flag = 0, 0
	
	# 将评论数少的日期合并，至少需要thres_count为一段
	reviews_partitions = []
	while flag<len(p_reviews)-1:
		tmp_count += reviews_count[flag]
		if tmp_count<thres_count and reviews_count[flag+1]<thres_count:
			flag += 2
		else:
			if tmp_count<thres_count and reviews_count[flag+1]>=thres_count:
				flag += 2
			else:
				flag += 1
			tmp_end_flag = flag-1
			reviews_partitions.append((tmp_start_flag, tmp_end_flag))
			tmp_start_flag = flag
			tmp_count = 0

	def merge_simis(reviews_partitions, max_text_length=5000):
		'''
		基于feature_words，考察评论段之间的相似性，得到simi_scores，并合并相似评论段
		param: reviews_partitions[list]: [(start_date_flag, end_date_flag),...]
		param: thres_simi: 用于合并评论的阈值
		return: new_reviews_partitions
		'''
		if len(reviews_partitions)==1:
			return reviews_partitions

		d_reviews_partitions = {}
		for i in range(len(reviews_partitions)):
			text = ""
			start, end = reviews_partitions[i]
			for j in range(start, end+1):
				text += p_reviews[j][1]
			feature_words = tags_extractor(text[:max_text_length], topk=topk, stops_sup=stops_sup)
			d_reviews_partitions[reviews_partitions[i]] = feature_words
		items = list(d_reviews_partitions.items())
		simi_scores = [0]
		for i in range(1, len(d_reviews_partitions)):
			simi_scores.append(words_simi_score(items[i][1], items[i-1][1], w2v_model))

		# 如果simi_score<thres_simi（和上一段内容不相似），则另起一段，否则合并为一段
		new_reviews_partitions = []
		i, j = 0, 1
		while j<len(items):
			while j<len(items) and simi_scores[j]>=thres_simi: j+=1
			new_reviews_partitions.append((items[i][0][0], items[j-1][0][1]))
			i = j
			j += 1

		return new_reviews_partitions

	for i in range(merge_num):
		reviews_partitions = merge_simis(reviews_partitions)

	d_reviews_partitions = {} # 将(start_date, end_date)作为key，feature_words作为value
	for i in range(len(reviews_partitions)):
		start, end = reviews_partitions[i]
		text = ""
		i = 0
		while p_reviews[i][0]<dates[start]: i+=1
		while p_reviews[i][0]<=dates[end]:
			text += p_reviews[i][1]
			i += 1
		d_reviews_partitions[(dates[start], dates[end])] = tags_extractor(text, topk=topk, stops_sup=stops_sup)

	# for k in d_reviews_partitions:
	# 	print(k, d_reviews_partitions[k])	

	return d_reviews_partitions


def check_breakouts():
	conn = MyConn()
	w2v_model = Word2Vec.load("../models/w2v/c3.mod")
	tracks = conn.query(sql="SELECT track_id, json_path FROM sub_tracks WHERE bnum>0")

	for track_id, filepath in tracks[70:]:
		d_reviews_partitions = get_reviews_partitions(filepath, w2v_model, merge_num=2)
		# print(d_reviews_partitions)
		breakouts = conn.query(table="breakouts", targets=["date", "reviews_num"], 
						conditions={"track_id":track_id, "release_drive":0, "fake":0, "capital_drive":0})
		if not breakouts: continue

		d_bcount = dict(zip(d_reviews_partitions.keys(), [0]*len(d_reviews_partitions)))
		for dt, reviews_num in breakouts:
			date = datetime.strftime(dt,'%Y-%m-%d')
			for k in d_reviews_partitions:
				if k[0]<=date and date<=k[1]:
					d_bcount[k] += 1
					break
		print(track_id)
		for k,v in d_bcount.items():
			if v>0:
				print("{} - {}: {} [count: {}]".format(k[0], k[1], 
							d_reviews_partitions[k], d_bcount[k]))
	




if __name__ == '__main__':
	w2v_model = w2v_model1 = Word2Vec.load("../models/w2v/c3.mod") 
	filepath = "/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews/1/71/1296410418.json" 
	# get_reviews_partitions(filepath, w2v_model=w2v_model)
	check_breakouts()


