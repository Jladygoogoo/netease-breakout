import os
import json
import numpy as np 
import pickle
from gensim.models import Word2Vec

import hdbscan


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





def main():
	w2v_path = "../models/w2v/b1.mod"
	dbscan_dir = "../models/dbscan/"
	data_path = "../data/breakouts-u2.json"

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

	# tags_pool = []
	# tags_set = set()
	# for item in data: 
	# 	tags_pool.extend(item["feature_words"])
	# 	tags_set.update(item["feature_words"])
	# tags_set = list(tags_set)
	# tags_vec1 = [w2v_model.wv.__getitem__(t) for t in tags_pool]
	# tags_vec2 = [w2v_model.wv.__getitem__(t) for t in tags_set]
	# print(len(tags_vec1), len(tags_vec2))


if __name__ == '__main__':
	main()







