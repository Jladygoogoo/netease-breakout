import os
import json
import numpy as np 
import pickle
from collections import Counter
from gensim.models import Word2Vec

import hdbscan

from connect_db import MyConn


def hdbscan_model(tags, w2v_model, cluster_selection_epsilon, min_cluster_size):
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


class TagsCluster:
	'''
	+ 话题组件
	+ 由多个相近的词组成，出现最多的词为话题中心（注意并不使用距离定义中心）
	+ tags: Counter，记录各个词的频率
	'''
	def __init__(self, tags):
			self.tags = Counter(tags)

	def __getattr__(self, name):
		if name=="center":
			return self.tags.most_common(1)[0][0]
		elif name=="size":
			return sum(self.tags.values())	

	def __add__(self, other):
		new_c = TagsCluster()
		new_c.tags = self.tags + other.tags 
		return new_c

	def similarity(self, tag, model):
		if not model.wv.__contains__(tag):
			return 0
		if tag in self.tags:
			return 1
		else:
			center = self.center
			simi = model.wv.similarity(center, tag)
			return simi

	def add(self, tag):
		self.tags.update(Counter([tag]))



class ClustersSet:
	'''
	+ 聚类组件
	+ 定义了聚类方法
		- growing, pruning, merging, classify(predict)
	'''
	def __init__(self, w2v_path, affinity=0.7, min_start_size=3):
		self.model = Word2Vec.load(w2v_path)
		self.affinity = affinity
		self.min_start_size = min_start_size
		self.input_tags_num = 0
		self.size = 0
		self.clusters = []


	def pruning(self):
		self.clusters = list(filter(lambda x:x.size>self.min_start_size, self.clusters))

	def merging(self):
		merge_num = 0
		roundd = len(self.clusters)
		while roundd>0:
			mergeflag = 0
			current_c = self.clusters[0]
			for j in range(1,len(self.clusters)):
				want_merge = 0
				for t1 in current_c.tags:
					for t2 in self.clusters[j].tags:
						if self.model.wv.similarity(t1,t2)>=self.affinity:
							want_merge += 1
				want_merge /= (len(current_c.tags)*len(self.clusters[j].tags))

				if want_merge>=self.affinity*0.5:
					new_cluster = current_c + self.clusters[j]
					self.clusters.pop(j)
					self.clusters.pop(0)
					self.clusters.append(new_cluster)
					merge_num += 1
					roundd -= 2
					mergeflag = 1
					break

			if not mergeflag:
				self.clusters.append(self.clusters.pop(0))
				roundd -= 1
		print("merging finished, {} cluster(s) being merged.".format(merge_num))


	def growing(self, tags, save_path, save_result=True):
		'''
		聚类生成主函数。
		不给定聚类数目，无法随机初始化，使用逐渐生长的方式。
		'''
		start = 0 
		# 避免w2v模型中不包含改词而报错
		while not self.model.wv.__contains__(tags[start]):
			start += 1
		self.clusters.append(TagsCluster(tags=[tags[start]]))
		self.input_tags_num += 1 
		
		for t in tags[start:]:
			if not self.model.wv.__contains__(t): 
				continue
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
				self.pruning()
				self.size = len(self.clusters)
				print("loading {} tags with {} clusters (add {})"\
						.format(self.input_tags_num, self.size, self.size-old_size))
				# 越到后期，剪枝的程度越大（min_size越大）
				self.min_start_size += 1

		self.pruning()
		# self.merging()
		self.clusters = sorted(self.clusters, key=lambda x:x.size, reverse=True)

		if save_result:
			self.save_clusters_result(save_path=save_path)



	def save_clusters_result(self, save_path):
		with open(save_path,'w') as f:
			for i,c in enumerate(self.clusters, start=1):
				tags = [p[0] for p in c.tags.most_common()]
				f.write("{} size:{} - [{}]".format(i,c.size,', '.join(tags))+'\n')


	def save(self, save_path):
		if not os.path.exists(os.path.dirname(save_path)):
			os.makedirs(os.path.dirname(save_path))

		with open(save_path,'wb') as f:
			try:
				pickle.dump(self, f)
				print("successfully saved clusters set.")
			except Exception as e:
				print("failed to save clusters set.")
				print("ERROR:", e)

	@classmethod
	def load(cls, load_path):
		with open(load_path,'rb') as f:
			print("loading {} object from {}".format(cls.__name__,load_path))
			obj = pickle.load(f)
			print("successfully loaded.")
		return obj


	def classify(self, tags):
		'''
		基于当前模型，对一组tag进行分类。
		使用预训练的tf-idf模型 
		'''
		target_clusters = []
		# 每一个tag都选择一个cluster
		for tt in tags:
			if not self.model.wv.__contains__(tt): continue
			clusters_max_simi = []
			max_simi = 0
			for i,c in enumerate(self.clusters):
				focus_range = min(5,len(c.tags))
				cluster_tags = list(zip(*c.tags.most_common()))[0][:focus_range]
				for ct in cluster_tags:
					simi = self.model.wv.similarity(tt,ct)
					if simi > max_simi:
						max_simi = simi
						# max_simi_tags = (tt,ct)
				clusters_max_simi.append(max_simi)
			if max(clusters_max_simi)<0.7:
				continue
			else:
				index = np.argmax(clusters_max_simi)
				cluster_center = self.clusters[index].center
				simi = max_simi
				target_clusters.append((index+1,cluster_center,simi))

		target_clusters = sorted(target_clusters, key=lambda x:x[2], reverse=True)
		unique_target_clusters = set()
		for c in target_clusters:
			c = c[:2]
			unique_target_clusters.add(c)

		if len(unique_target_clusters)==0: 
			unique_target_clusters.add((0,'others'))
		return unique_target_clusters



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
		show_info = lambda x:("{}-{}".format(x[0]+1,clusters_set.clusters[int(x[0])].center),x[1]) if x[0]!='others' else x
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
	w2v_path = "../models/w2v/b1.mod"
	my_cluster_save_path = "../models/my_cluster/my_cluster_1.pkl"
	res_save_path = "../records/my_cluster_1_res.txt"

	feature_words_packed = [r[0].split() for r in conn.query(targets=["feature_words"], table="breakouts_feature_words_1")] + \
				[r[0].split() for r in conn.query(targets=["feature_words"], table="no_breakouts_feature_words_1")]
	feature_words = []
	for p in feature_words_packed: feature_words.extend(p)

	# print(len(feature_words)) # 205500
	# print(len(set(feature_words))) # 15648
	
	my_cluster = ClustersSet(w2v_path=w2v_path, affinity=0.65)	
	my_cluster.growing(feature_words, save_path=res_save_path)
	my_cluster.save(save_path=my_cluster_save_path)


if __name__ == '__main__':
	test_my_cluster()






