import os
import json
import pandas as pd
import pickle
import traceback
import random
from threading import Thread, Lock
from multiprocessing import Lock as PLock
from queue import Queue
import logging

import librosa
from gensim.models import Word2Vec

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from breakout_tools import get_reviews_df, get_reviews_count, get_breakouts, get_breakouts_text
from utils import assign_dir, get_tracks_set_db, get_dir_item_set
from preprocess import tags_extractor


def get_from_db(track_id, targets):
	conn = MyConn()
	res = conn.query(targets=targets, conditions={"track_id": track_id})
	return list(res[0])



def get_main_tagged_tracks(r_path="../data/pucha/BorgCube2_65b1/BorgCube2_65b1.csv"):
	'''
	提取出指定聚类关键词的歌曲。
	'''
	main_tag_clusters_d = {"短视频":[1,44], "微博":[2,18], "高考":[3,22,35], "节日":[8,9,38,34]}

	main_tagged_tracks = set()
	main_tagged_tracks_d = {}
	for k in main_tag_clusters_d:
		main_tagged_tracks_d[k] = set()
	
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
	'''
	获取未爆发歌曲
	'''
	sql = 'SELECT track_id FROM tracks WHERE first_review BETWEEN %s AND %s and reviews_num > %s and have_mp3=1 and have_lyrics=1'
	
	# 筛选条件：
	# 至少在一年以前发布，至少有1000条评论的歌曲
	candidates = get_tracks_set_db(sql=sql, conditions={"first_review1":"2013-01-01", "first_review2":"2019-02-01",
											 "reviews_num":1000})

	breakouts_tracks = set(map(str, pd.read_json("../data/breakouts-0.json")["track_id"].unique()))

	no_breakouts_tracks = candidates - candidates.intersection(breakouts_tracks)
	print(len(no_breakouts_tracks))
	no_breakouts_tracks = random.sample(no_breakouts_tracks, 7000)

	with open("../data/no_breakouts_tracks.txt",'w') as f:
		f.write('\n'.join(no_breakouts_tracks))


def extract_raw_music_data(tracks_set, prefix, flag0=0):
	'''
	开启多线程保存rawmusic文件
	params:
		tracks_set: 歌曲集
		prefix: 保存路径头
		flag: 起始编号
	'''
	# 单个thread的任务
	def task(thread_id, task_args):
		conn = MyConn()
		while not task_args["pool"].empty():
			track_id = task_args["pool"].get()

			mp3_path = "/Volumes/nmusic/NetEase2020/data" + \
				conn.query(targets=["mp3_path"], conditions={"track_id": track_id})[0][0]

			try:
				y, sr = librosa.load(mp3_path, duration=30, offset=10)
				if librosa.get_duration(y, sr)<30:
					print("track-{} is shorter than 40s".format(track_id))
					continue
				# 存储路径
				pkl_path = os.path.join(
					assign_dir(prefix=task_args["prefix"], 
						n_dir=task_args["n_dir"], 
						dir_size=task_args["dir_size"], 
						flag=task_args["flag"]),
					"{}.pkl".format(track_id)
				)
				# 将flag+1以更新存储路径，需要lock
				task_args["lock"].acquire()
				task_args["flag"] += 1
				task_args["lock"].release()

				if not os.path.exists(os.path.dirname(pkl_path)):
					os.makedirs(os.path.dirname(pkl_path))
				with open(pkl_path, 'wb') as f:
					pickle.dump(y, f)
				# print(thread_id, pkl_path)

			except:
				print("track-{} failed to process.".format(track_id))

	if input("flag={}，确定请输入y: ".format(flag0)) != 'y':
		print("你个傻子😐")
		return

	# 开启thread群工作
	lock = Lock()
	pool = Queue()
	flag = flag0
	for t in tracks_set: 
		pool.put(t)
		
	task_args = {"lock":lock, "pool":pool,
		"n_dir":2, "dir_size":(10, 100), "flag": flag, "prefix": prefix}
	threads_group = ThreadsGroup(task=task, n_thread=10, task_args=task_args)
	threads_group.start()


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



def add_feature_words_to_db():
	'''
	直接在数据库中更新feature_words，使用多线程
	'''
	def task(pid, task_args):
		conn = MyConn()
		w2v_model = Word2Vec.load("../models/w2v/b1.mod")
		while 1:
			task_args["lock"].acquire()
			res = conn.query(targets=["breakout_id", "text_path"], conditions={"have_words":0}, 
					table="breakouts", fetchall=False)
			if res is not None:
				breakout_id, text_path = res
				conn.update(table="breakouts",
							settings={"have_words": 1},
							conditions={"breakout_id": breakout_id})
				task_args["lock"].release()

				try:
					feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
					conn.insert(table="breakouts_feature_words_1",
								settings={"breakout_id":breakout_id, "feature_words":" ".join(feature_words)})

					print("[Process-{}] breakout_id: {}, feature_words: {}".format(pid, breakout_id, feature_words))
				except:
					conn.update(table="breakouts",
								settings={"have_words": 0},
								conditions={"breakout_id": breakout_id})
					break

			else:
				task_args["lock"].release()
				break

	lock = PLock()
	task_args = {"lock":lock}
	process_group = ProcessGroup(task=task, n_procs=5, task_args=task_args)
	process_group.start()



def extract_no_breakouts_feature_words():
	def task(pid, task_args):
		conn = MyConn()
		w2v_model = Word2Vec.load("../models/w2v/b1.mod")
		while 1:
			track_id = task_args["queue"].get()
			text_path = 
			try:
				feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
				conn.insert(table="breakouts_feature_words_1",
							settings={"breakout_id":breakout_id, "feature_words":" ".join(feature_words)})

				print("[Process-{}] breakout_id: {}, feature_words: {}".format(pid, breakout_id, feature_words))
			except:
				conn.update(table="breakouts",
							settings={"have_words": 0},
							conditions={"breakout_id": breakout_id})
				break

			else:
				task_args["lock"].release()
				break
	no_breakouts_tracks = open("../data/no_breakouts_tracks.txt").read().splitlines()
	queue = PQueue()
	for t in no_breakouts_tracks:
		queue.put(t)



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
	path = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
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
	
	# prefix = "/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic"
	# path = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
	# tracks_set = set(map(str, pd.read_json("../data/breakouts-0.json")["track_id"].unique()))
	# existed = get_dir_item_set(path, file_postfix=".pkl")
	# tracks_set = tracks_set - existed
	# print(len(tracks_set))
	# extract_raw_music_data(tracks_set, prefix=path, flag0=6586)


	# no_breakouts_tracks = open("../data/no_breakouts_tracks.txt").read().splitlines()
	# extract_raw_music_data(no_breakouts_tracks, 
	# 			prefix="/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic",
	#  			flag0=0)

	add_feature_words_to_db()
