import os
import json
import pandas as pd
import pickle
import traceback
import random
from threading import Thread, Lock
from queue import Queue
import logging

import librosa
from gensim.models import Word2Vec

from connect_db import MyConn
from threads import ThreadsGroup
from breakout_tools import get_reviews_df, get_reviews_count, get_breakouts, get_breakouts_text
from utils import assign_dir
from preprocess import tags_extractor


def get_from_db(track_id, targets):
	conn = MyConn()
	res = conn.query(targets=targets, conditions={"track_id": track_id})
	return list(res[0])


def extract_raw_music_data(tracks_set, w_dir):
	# 单个thread的任务
	def task(thread_id, task_args):
		logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S', filename=task_args["log_file"])

		while not task_args["pool"].empty():
			track_id = task_args["pool"].get()
			mp3_path = "/Volumes/nmusic/NetEase2020/data" + get_from_db(track_id=track_id, targets=["mp3_path"])[0]

			try:
				y, sr = librosa.load(mp3_path, duration=30, offset=10)
				if librosa.get_duration(y, sr)<30:
					print("track-{} is shorter than 40s".format(track_id))
					continue

				task_args["log_lock"].acquire()
				n_dir = task_args["flag"] // 100
				task_args["flag"] += 1
				logging.info("[Thread-{}] successfully load track-{}, flag = {}".format(thread_id, track_id, task_args["flag"]-1))
				task_args["log_lock"].release()

				w_path = task_args["write_dir"] + "{}/{}.pkl".format(n_dir, track_id)
				if not os.path.exists(os.path.dirname(w_path)):
					os.makedirs(os.path.dirname(w_path))
				with open(w_path, 'wb') as f:
					pickle.dump(y, f)

			except:
				task_args["log_lock"].acquire()
				logging.info("[Thread-{}] failed to load track-{}".format(thread_id, track_id))
				task_args["log_lock"].release()

	# 开启thread群工作
	log_lock = Lock()
	log_file = "../logs/extract_raw_music_data.log"
	pool = Queue()
	for t in tracks_set: 
		pool.put(t)

	task_args = {"log_lock":log_lock, "log_file":log_file, 
				"pool":pool, "flag":1754,
				"write_dir": w_dir}
	threads_group = ThreadsGroup(task=task, n_thread=10, task_args=task_args)
	threads_group.start()



def get_main_tagged_tracks():
	main_tag_clusters_d = {"短视频":[1,44], "微博":[2,18], "高考":[3,22,35], "节日":[8,9,38,34]}

	main_tagged_tracks = set()
	main_tagged_tracks_d = {}
	for k in main_tag_clusters_d:
		main_tagged_tracks_d[k] = set()

	r_path = "../data/pucha/BorgCube2_65b1/BorgCube2_65b1.csv"
	df = pd.read_csv(r_path)

	for _, row in df.iterrows():
		track_id = int(row["file"][:-5])
		clusters = eval(row["cluster_number"])
		for c in clusters:
			for k,v in main_tag_clusters_d.items():
				if c[0] in v:
					main_tagged_tracks.add(track_id)
					main_tagged_tracks_d[k].add(track_id)

	print("total:", len(main_tagged_tracks))
	count = 0
	for k, v in main_tagged_tracks_d.items():
		count += len(v)
		print(k, len(v))

	# 统计cluster交叉
	def calc_cross_rate(set1, set2):
		inter_set = set1.intersection(set2)
		union_set = set1.union(set2)
		return len(inter_set)/len(union_set)

	# 统计cluster交叉
	def cross_view():
		# 注释掉的内容是用来画图的
		main_tag_clusters = list(main_tag_clusters_d.keys())
		cross_matrix = []
		for i in range(len(main_tag_clusters)):
			tmp = []
			for j in range(i+1, len(main_tag_clusters)):
				tc1, tc2 = main_tag_clusters[i], main_tag_clusters[j]
				print("{}-{}: {:.3f}%".format(tc1, tc2, 100*calc_cross_rate(main_tagged_tracks_d[tc1], 
																		main_tagged_tracks_d[tc2])))


	conn = MyConn()
	res = conn.query(targets=["track_id"], conditions={"have_lyrics":1, "have_mp3":1})
	set2 = set()
	for r in res:
		set2.add(r[0])
	u_set = set2.intersection(main_tagged_tracks)
	for label, tracks in main_tagged_tracks_d.items():
		main_tagged_tracks_d[label] = set(tracks).intersection(set2)


	# 将 valid 歌曲id 写出
	# with open("../results/tracks_set/main_tagged_tracks.txt", 'w') as f:
	# 	f.write('\n'.join(map(str,u_set)))

	with open("../data/main_tagged_tracks/labels_dict.pkl", 'wb') as f:
		pickle.dump(main_tagged_tracks_d, f)




def get_no_breakouts_tracks():
	sql1 = 'SELECT track_id FROM tracks WHERE first_review BETWEEN %s AND %s and reviews_num BETWEEN %s AND %s and have_mp3=1 and have_lyrics=1'
	# sql2 = 'SELECT track_id FROM tracks WHERE first_review BETWEEN %s AND %s and reviews_num BETWEEN %s AND %s have_mp3=1 and have_lyrics=1'
	
	# res = get_from_db(sql=sql)
	conn = MyConn()
	res1 = conn.query(sql=sql1, conditions={"first_review1":"2013-01-01", "first_review2":"2014-01-01",
											 "reviews_num1":1000, "reviews_num2":10000})
	res1 = set([str(r[0]) for r in res1])
	res2 = conn.query(sql=sql1, conditions={"first_review1":"2018-01-01", "first_review2":"2019-01-01",
											 "reviews_num1":1000, "reviews_num2":10000})
	res2 = set([str(r[0]) for r in res2])
	res3 = conn.query(sql=sql1, conditions={"first_review1":"2014-01-01", "first_review2":"2018-01-01",
											 "reviews_num1":1000, "reviews_num2":10000})
	res3 = set([str(r[0]) for r in res3])


	# breakouts_tracks = pd.read_json("../data/breakouts-u2.json")["track_id"].unique()
	breakouts_tracks = pd.read_csv("../data/pucha/BorgCube2_65b1/BorgCube2_65b1.csv")["file"].unique()
	# breakouts_tracks = set(map(lambda x:x[:-5], list(breakouts_tracks))) + set(pd.read_json("../data/breakouts-u2.json")["track_id"].unique())
	print(len(breakouts_tracks))

	no_breakouts_tracks1 = res1 - res1.intersection(breakouts_tracks)
	no_breakouts_tracks2 = res2 - res2.intersection(breakouts_tracks)
	no_breakouts_tracks3 = res3 - res3.intersection(breakouts_tracks)
	print(len(no_breakouts_tracks1), len(no_breakouts_tracks2), len(no_breakouts_tracks3))
	no_breakouts_tracks = random.sample(no_breakouts_tracks1, 700) + random.sample(no_breakouts_tracks2, 700) + random.sample(no_breakouts_tracks2, 1600)
	print(len(no_breakouts_tracks))

	with open("../data/no_breakouts_tracks.txt",'w') as f:
		f.write('\n'.join(no_breakouts_tracks))


def get_breakouts_json():
	path = "/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews"
	json_path = "../data/breakouts-0.json"
	w_prefix = "/Volumes/nmusic/NetEase2020/data/breakouts_text/breakouts-0"
	n_dir = 2
	dir_size = (100, 100)

	# 把没有lyrics和mp3的去除
	conn = MyConn()
	res = conn.query(targets=["track_id"], conditions={"have_lyrics":1, "have_mp3":1})
	backup_tracks = set()
	for r in res:
		backup_tracks.add(str(r[0]))

	write_content = []
	total_count = 0
	for root,dirs,files in os.walk(path):
		for file in files:
			n = int(root.split('/')[-2])*100 + int(root.split('/')[-1])
			if "DS" in file or file[:-5] not in backup_tracks:
				continue
			try:
				track_id = file[:-5]
				filepath = os.path.join(root, file)
				df = get_reviews_df(filepath)
				reviews_count, dates =  get_reviews_count(df["date"].values)

				breakouts_group = get_breakouts(reviews_count, min_reviews=200)
				breakouts = [g[0] for g in breakouts_group]
				bdates = [dates[p[0]] for p in breakouts]

				for i, p in enumerate(breakouts):
					beta = p[1]
					date = bdates[i]
					reviews_num = reviews_count[p[0]]
					btext = get_breakouts_text(df, date)
					w_path = os.path.join(
						assign_dir(prefix=w_prefix, n_dir=n_dir, dir_size=dir_size, flag=total_count),
						"{}-{}.txt".format(track_id, i)
					)
					# with open(w_path, 'w') as f:
					# 	f.write(btext)
					write_content.append({
						"track_id": track_id,
						"flag": i,
						"beta": beta,
						"date": date,
						"reviews_num": reviews_num,
						"text_path": w_path
					})
					total_count += 1
					if total_count % 100 == 0:
						print("total_count = {}".format(total_count))
						if total_count==200:
							with open("../data/breakouts-00.json", 'w') as f:
								json.dump(write_content, f, ensure_ascii=False, indent=2)
			except:
				print(traceback.format_exc())
				return

	with open(json_path, 'w') as f:
		json.dump(write_content, f, ensure_ascii=False, indent=2)


def add_feature_words_to_json():
	'''
	基于text_path读取文本加入feature_words属性。
	param: json_data_list: 格式参见 "../data/breakouts-0.json"
	return: new_json_data_list: 新增feature_words属性后的数据
	example:
		item = {
		    "track_id": "4872534",
		    "flag": 0,
		    "beta": 394.4,
		    "date": "2018-03-05",
		    "reviews_num": 454,
		    "text_path": "/Volumes/nmusic/NetEase2020/data/breakouts_text/breakouts-0/0/0/4872534-0.txt",
		    "feature_words": ["打卡","机械","材料","成型","下课","可爱","铃声","学院","学校","下课铃"]
		}
	'''
	w2v_model = Word2Vec.load("/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod")
	for i in range(len(data)):
		try:
			text_path = data[i]["text_path"]
			feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
			data[i]["feature_words"] = feature_words
			print(i, data[i]["track_id"], feature_words)
		except KeyboardInterrupt:
			with open("../data/breakouts-3.json",'w') as f:
				print(i)
				new_data = data[:i]
				json.dump(new_data, f, ensure_ascii=False, indent=2)	
			return
		except:
			print(i, data[i]["track_id"],"ERROR")

	with open("../data/breakouts-3.json",'w') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)	


def prep_rawmusic_data(tracks_set, prefix, flag):
	'''
	提取rawmusic数据并保存
	param:
		tracks_set: track_id集合
		prefix: 保存路径头
	'''
	conn = MyConn()
	
	# flag = 2000
	n_dir = 2
	dir_size = (10, 100)
	for t in tracks_set:
		try:
			mp3_path = "/Volumes/nmusic/NetEase2020/data" + conn.query(targets=["mp3_path"], conditions={"track_id":t})[0][0]
			y, sr = librosa.load(mp3_path, duration=30, offset=10)
			if librosa.get_duration(y, sr)<30:
				print("track-{} is shorter than 40s".format(t))
				continue
			pkl_path = os.path.join(
				assign_dir(prefix=prefix, n_dir=n_dir, dir_size=dir_size, flag=flag),
				"{}.pkl".format(t)
			)
			with open(pkl_path, 'wb') as f:
				pickle.dump(y, f)
			flag += 1
			# conn.update(settings={"rawmusic_path":pkl_path}, conditions={"track_id":t})
			print(flag, t)
		except KeyboardInterrupt:
			return
		except:
			print(flag, t, "ERROR")
			print(traceback.format_exc())



def mymodel_data():
	'''
	完善 "../data/breakouts-0.json" 保存爆发点的数据，以供MyModel使用
	+ 添加 feature_words
	+ 提取 rawmusic_data
	'''

	json_path = "../data/breakouts-0.json"	
	with open(json_path) as f:
		content = json.load(f)
	data = content[3666:5000]

	add_feature_words_to_json(data)
	tracks_set = set([p["track_id"] for p in data])
	prefix = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
	flag = 2000
	prep_rawmusic_data(tracks_set, prefix, flag)






'''
===============================
以下均是一些杂碎的不值得拥有姓名的操作
===============================
'''

def update_rawmusic_path():
	conn = MyConn()
	path = "/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic"
	for root, dirs, files in os.walk(path):
		for file in files:
			if "DS" in file: continue
			track_id = file[:-4]
			pkl_path = os.path.join(root, file)
			try:
				conn.update(settings={"rawmusic_path":pkl_path}, conditions={"track_id":track_id})
			except:
				print("[ERROR]: {}".format(track_id))


def regroup_json():
	w_path = "../data/breakouts-u2.json"
	r_path1 = "../data/breakouts-u1.json"
	r_path2 = "../data/breakouts-3.json"

	with open(r_path1) as f1, open(r_path2) as f2:
		content1 = json.load(f1)
		content2 = json.load(f2)

	valid_rawmusic_tracks = set()
	for root, dirs, files in os.walk("/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"):
		for file in files:
			if "DS" in file: continue
			valid_rawmusic_tracks.add(file[:-4])

	unique_items = set()
	
	content3 = []
	for item in content1:
		if (item["track_id"],item["flag"]) in unique_items or item["track_id"] not in valid_rawmusic_tracks:
			continue
		else:
			unique_items.add((item["track_id"],item["flag"]))
			content3.append(item)

	for item in content2:
		if (item["track_id"],item["flag"]) in unique_items or item["track_id"] not in valid_rawmusic_tracks:
			continue
		else:
			unique_items.add((item["track_id"],item["flag"]))
			content3.append(item)

	# print(len(content3))
	with open("../data/breakouts-u2.json", 'w') as f:
		json.dump(content3, f, ensure_ascii=False, indent=2)




def mymodel_test_data():
	json_path = "../data/breakouts-00.json"
	with open(json_path) as f:
		content = json.load(f)
	test_data = content[:10]

	w2v_model = Word2Vec.load("/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod")
	for i in range(len(test_data)):
		text_path = test_data[i]["text_path"]
		feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
		test_data[i]["feature_words"] = feature_words
		print(i, test_data[i]["track_id"], feature_words)

	with open("../data/test/breakouts-test0.json",'w') as f:
		json.dump(test_data, f)

	# conn = MyConn()
	# tracks_set = set([p["track_id"] for p in test_data])
	# for t in tracks_set:
	# 	mp3_path = "/Volumes/nmusic/NetEase2020/data" + conn.query(targets=["mp3_path"], conditions={"track_id":t})[0][0]
	# 	y, sr = librosa.load(mp3_path, duration=30, offset=10)
	# 	if librosa.get_duration(y, sr)<30:
	# 		print("track-{} is shorter than 40s".format(t))
	# 		continue
	# 	pkl_path = "../data/test/rawmusic/0/0/{}.pkl".format(t)
	# 	with open(pkl_path, 'wb') as f:
	# 		pickle.dump(y, f)
	# 	conn.update(settings={"rawmusic_path":pkl_path}, conditions={"track_id":t})
	# 	print(t)


if __name__ == '__main__':
	# get_breakouts_json()
	# mymodel_test_data()
	# mymodel_data()
	# update_rawmusic_path()
	# regroup_json()
	# get_no_breakouts_tracks()
	# no_breakouts_tracks = open("../data/no_breakouts_tracks.txt").read().splitlines()
	# prefix = "/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic"
	# prep_rawmusic_data(no_breakouts_tracks, prefix, flag=0)

	with open("../data/mymodel_data/dataset_embed-4-1000.pkl", "rb") as f:
		content = pickle.load(f)
	print(content[0][0] - content[0][1])
	# print(content[0][1])
	# print(content[0][-2])
	# print(content[0][-1])
	# 