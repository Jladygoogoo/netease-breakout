import os
import json
import numpy as np 
import pandas as pd
import pickle
from collections import Counter
from gensim.models import Word2Vec

import hdbscan

from connect_db import MyConn
from w2v import words_simi_score


def hdbscan_model(tags, w2v_model, cluster_selection_epsilon, min_cluster_size):
	'''
	dbscan聚类模型。
	'''

	# 得到词向量表示
	tags_vec = [w2v_model.wv.__getitem__(t) for t in tags]

	# 模型创建与训练
	dbscan_model = hdbscan.HDBSCAN(
		cluster_selection_epsilon=cluster_selection_epsilon,
		min_cluster_size=min_cluster_size, 
		prediction_data=True).fit(tags_vec)

	# 聚类结果（标签和可能性）
	labels, probs = dbscan_model.labels_, dbscan_model.probabilities_

	# 聚类类别数目
	print(len(set(labels)))
	clusters = {}
	for i in range(len(labels)):
		if labels[i] not in clusters:
			clusters[labels[i]] = [(tags[i], probs[i])]
		else:
			clusters[labels[i]].append((tags[i], probs[i]))

	for c in clusters:
		clusters[c] = list(zip(*sorted(clusters[c], key=lambda p:p[1], reverse=True)))[0]

	for k, v in clusters.items():
		print(k, v)

	# with open(os.path.join(dbscan_dir, "dbscan-1.pkl"), "wb") as f:
	# 	pickle.dump(dbscan_model, f)


class Tag:
	'''
	标签组件。
	包含标签的内容、所属爆发的时间、规模。
	'''
	def __init__(self, b_text, b_date, b_size):
		self.b_text = b_text #str
		self.b_date = b_date # str
		self.b_size = b_size # int



class TagsCluster:
	'''
	自定义聚类簇组件。
	+ 由多个相近的词组成，出现最多的词为话题中心（注意并不使用距离定义中心）
	+ b_texts_counter: Counter，记录各个词的频率
	'''
	def __init__(self, tags):
		# 保存Tag对象，提取出b_text用于合并
		b_texts = []
		self.tags = tags 
		for t in tags:
			b_texts.append(t.b_text)
		self.b_texts_counter = Counter(b_texts)


	def __getattr__(self, attr):
		if attr=="center_text":
			return self.b_texts_counter.most_common(1)[0][0]
		elif attr=="cluster_size":
			return len(self.tags)
		elif attr=="avg_breakout_size":
			return np.mean([t.b_size for t in self.tags])
		elif attr=="std_breakout_size":
			return np.std([t.b_size for t in self.tags])


	def __add__(self, other):
		tags = self.tags + other.tags
		new_c = TagsCluster(tags)
		return new_c


	def similarity(self, tag, model):
		if not model.wv.__contains__(tag.b_text):
			return 0
		if tag.b_text in self.b_texts_counter:
			return 1
		else:
			center_text = self.center_text
			simi = model.wv.similarity(center_text, tag.b_text)
			return simi

	def add(self, tag):
		self.tags.append(tag)
		self.b_texts_counter.update(Counter([tag.b_text]))



class ClustersSet:
	'''
	自定义聚类模型。
	methods:
		+ growing：聚类过程
		+ pruning：剪枝
		+ merging：合并相似聚类组建
		+ classify(predict)：预测
	'''
	def __init__(self, w2v_path, affinity=0.7, min_start_size=3):
		self.model = Word2Vec.load(w2v_path)
		self.affinity = affinity
		self.min_start_size = min_start_size
		self.input_tags_num = 0
		self.size = 0
		self.clusters = []


	def grow(self, tags):
		'''
		聚类生成主函数。
		不给定聚类数目，无法随机初始化，使用逐渐生长的方式。
		'''
		start = 0 
		self.clusters.append(TagsCluster(tags=[tags[0]]))
		self.input_tags_num += 1 
		
		for t in tags[1:]:
			simis = [cluster.similarity(t, self.model) for cluster in self.clusters]

			hit_cluster_i = np.argmax(simis)
			# 相似度大于affinity则加入
			if simis[hit_cluster_i] > self.affinity:
				self.clusters[hit_cluster_i].add(t)
			# 否则创建新的话题组件
			else:
				self.clusters.append(TagsCluster(tags=[t]))

			self.input_tags_num += 1 
			# 定期进行剪枝（将size过小的话题组件删除）
			if self.input_tags_num%1000==0:
				old_size = self.size
				self.prune()
				self.size = len(self.clusters)
				print("loading {} tags with {} clusters (add {})"\
						.format(self.input_tags_num, self.size, self.size-old_size))
				# 越到后期，剪枝的程度越大（min_size越大）
				self.min_start_size += 1

		self.prune()
		self.merge()
		self.clusters = sorted(self.clusters, key=lambda x:x.cluster_size, reverse=True)


	def prune(self):
		'''
		剪枝。将包含tag较少的tag_cluster扔掉
		'''
		self.clusters = list(filter(lambda x:x.cluster_size>self.min_start_size, self.clusters))

	def merge(self):
		clusters_num = len(self.clusters)
		new_clusters = []

		for i in range(clusters_num-1, 0, -1):
			words1 = [p[0] for p in self.clusters[i].b_texts_counter.most_common(5)]
			merge_flag = 0
			for j in range(i):
				words2 = [p[0] for p in self.clusters[j].b_texts_counter.most_common(5)]
				if words_simi_score(words1, words2, self.model)>=0.65:
					self.clusters[j] += self.clusters[i]
					print("merging c{}-{} to c{}-{}".format(j, i, words1, words2))
					merge_flag = 1
					break
			if not merge_flag:
				new_clusters.append(self.clusters[i])

		self.clusters = new_clusters



	def save(self, model_path, txt_path=None, csv_path=None, bsizes_csv_path=None):
		'''
		model_path: 保存模型
		txt_path / csv_path: 保存聚类结果
		reviews_num_csv_path: 用于保存爆发规模的具体数据
		'''

		if txt_path or csv_path:
			self.save_clusters_result(txt_path, csv_path, bsizes_csv_path)

		with open(model_path,'wb') as f:
			try:
				pickle.dump(self, f)
				print("successfully saved clusters set.")
			except Exception as e:
				print("failed to save clusters set.")
				print("ERROR:", e)


	def save_clusters_result(self, txt_path=None, csv_path=None, bsizes_csv_path=None):
		'''
		保存聚类结果为文件。
		'''
		if txt_path:
			with open(txt_path,'w') as f:
				for i,c in enumerate(self.clusters, start=1):
					b_texts = [p[0] for p in c.b_texts_counter.most_common()]
					f.write("{} size:{} - [{}]".format(i,c.cluster_size,', '.join(b_texts))+'\n')
		if csv_path:
			data = [(c.center_text, " ".join([p[0] for p in c.b_texts_counter.most_common(5)]), 
					c.cluster_size, c.avg_breakout_size, c.std_breakout_size) for c in self.clusters]
			df = pd.DataFrame(data, columns=["center", "represents", "cluster_size", "avg_breakout_size", "std_breakout_size"])
			df.to_csv(csv_path, encoding="utf-8-sig")

		if bsizes_csv_path:
			data = []
			for c in self.clusters:
				center_text = c.center_text
				b_sizes = [t.b_size for t in c.tags]
				for b_size in b_sizes:
					data.append((center_text, b_size))
			df = pd.DataFrame(data, columns=["center_text", "b_size"])
			df.to_csv(bsizes_csv_path, encoding="utf-8-sig")


	@classmethod
	def load(cls, load_path):
		'''
		加载模型。
		'''
		with open(load_path,'rb') as f:
			print("loading {} object from {}".format(cls.__name__,load_path))
			obj = pickle.load(f)
			print("successfully loaded.")
		return obj


	def predict(self, text):
		'''
		对爆发文本进行归类。
		'''
		scores = []
		c_words_all = []
		for c in self.clusters:
			c_words = [p[0] for p in self.clusters[i].b_texts_counter(5)]
			scores.append(words_simi_score([text], c_words, self.model))
			c_words_all.append(c_words)
		if max(scores) > 0.55:
			print("word-{} belongs to c-{}".format(text, c_words_all[np.argmax[scores]]))



def reviews_num_in_clusters(clusters_set, image_save_dir):
	'''
	不同话题聚类中评论数量的分布
	'''
	if not os.path.exists(image_save_dir): os.makedirs(image_save_dir)
	data = []
	for i,c in enumerate(clusters_set.clusters,start=1):
		if c.size <= 15: continue
		beta0, std_sigma, avg = power_law_plot(
						c.reviews_nums, 
						title='cluster-{}'.format(i), 
						# save=True,
						save=False,
						save_path=os.path.join(image_save_dir,'cluster{}.png'.format(i))
					)
		if not beta0: continue
		# print('cluster-{}: beta0={:.2f}, std_sigma={:.2f}, avg={:.2f}'.\
		# 		format(i,beta0,std_sigma,avg))
		data.append([i,beta0,avg,c.size,std_sigma])

	sigma_thres = 0.7
	new_data = list(filter(lambda p:p[-1]<=sigma_thres,data))
	print("keep rate: {:.3f}%".format(len(new_data)/len(data) * 100))

	n_clusters = 6
	method = 'kmeans'
	double_cluster(new_data, n_clusters=n_clusters, method=method, 
			image_save_path='../results/clusters_in_clusters/2cluster_{}_{}_{}_513d.png'.format(method,n_clusters,sigma_thres))
		


def clusters_in_reviews_rank(clusters_set, ranks_list, image_save_path, show_tags_num=8):
	'''
	不同数量级评论中话题聚类的分布
	'''
	reviews_rank2cluster = {}

	for i,c in enumerate(clusters_set.clusters):
		for rn in c.reviews_nums:
			level = rn//100
			for j in range(len(ranks_list)):
				start,end = ranks_list[j]
				if level>=start and level<=end: 
					rank = j
					break

			if rank in reviews_rank2cluster:
				reviews_rank2cluster[rank].update([i])
			else:
				reviews_rank2cluster[rank] = Counter([i])


	reviews_rank2cluster = sorted(reviews_rank2cluster.items(),key=lambda x:x[0])
	all_ranks_tags = []
	for rank,counter in reviews_rank2cluster:
		rank_tags = sorted(counter.most_common(),key=lambda x:x[1],reverse=True)

		pruned_rank_tags = []
		if np.sum([x[1] for x in rank_tags])>20:
			rank_tags = list(filter(lambda x:x[1]>1,rank_tags))
		
		tail_num = np.sum([x[1] for x in rank_tags[show_tags_num:]])
		if tail_num>0:
			rank_tags = rank_tags[:show_tags_num] + [('others',float(tail_num))]
		show_info = lambda x:("{}-{}".format(x[0]+1,clusters_set.clusters[int(x[0])].center_text),x[1]) if x[0]!='others' else x
		rank_tags = list(map(show_info,rank_tags))

		all_ranks_tags.append(rank_tags)

	draw_donut(all_ranks_tags)
	# draw_heap(all_ranks_tags,ranks_list)


def test_dbscan():
	w2v_path = "../models/word2vec/b1.mod"
	dbscan_dir = "../models/dbscan/"

	cluster_selection_epsilon = 0.1 # eps
	min_cluster_size = 2 # min_samples

	with open(data_path) as f:
		data = json.load(f)

	w2v_model = Word2Vec.load(w2v_path)

	tags_set = set()
	for item in data: 
		tags_set.update(item["feature_words"])
	tags_set = list(tags_set)
	hdbscan_model(tags_set, w2v_model, cluster_selection_epsilon, min_cluster_size)


def test_my_cluster():
	conn = MyConn()
	w2v_path = "../models/w2v/c4.mod"
	rubbish_tags = open("../resources/rubbish_words_tags.txt").read().splitlines()
	w2v_model = Word2Vec.load(w2v_path)

	valid_breakouts = conn.query(sql="SELECT id, date, reviews_num FROM breakouts WHERE release_drive=0 AND capital_drive=0 AND fake=0")
	valid_breakouts_info_d = dict(zip([p[0] for p in valid_breakouts], [(p[1],p[2]) for p in valid_breakouts]))
	breakouts_id_tags_p = conn.query(table="breakouts_feature_words_c3", targets=["id","clean_feature_words"])

	tags_pool = []
	for id_, tags in breakouts_id_tags_p:
		if id_ in valid_breakouts_info_d:
			b_date, b_size = valid_breakouts_info_d[id_]
			for t in tags.split():
				if t not in rubbish_tags and w2v_model.wv.__contains__(t):
					tags_pool.append(Tag(t, b_date, b_size))

	print(len(tags_pool)) # 24796
	
	my_cluster = ClustersSet(w2v_path=w2v_path, affinity=0.55)	
	my_cluster.grow(tags_pool)
	my_cluster.save(model_path="../models/my_cluster/my_cluster_1.pkl", 
						txt_path="../results/my_cluster_1_res.txt", 
						csv_path="../results/my_cluster_1_res.csv",
						bsizes_csv_path="../results/clusters_bsizes.csv")
	# with open()


if __name__ == '__main__':
	test_my_cluster()






