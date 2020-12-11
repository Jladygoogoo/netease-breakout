import os
import numpy as np
import random
from gensim import corpora, models
# from gensim.models.tfidfmodel import TfidfModel
from connect_db import MyConn
from preprocess import tags_extractor


def build_tfidf_model(tracks_set):
	'''
	数据来源：breakout_tracks_set, no_breakout_tracks_set
	处理办法：每首歌至多抽取1000条评论（随机），取topk=20构建doc
	'''

	conn = MyConn()
	w2v_model = models.Word2Vec.load("/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod")
	files = []
	for track_id in tracks_set:
		files.append(conn.query(targets=["text_path"], conditions={"track_id": track_id}, fetchall=False)[0])

	docs = []
	for i, file in enumerate(files):
		print(i)
		# content = open(file).read()[:1000]
		content = open(file).read().splitlines()
		content = random.sample(content, min(100, len(content)))
		content = "\n".join(content)
		docs.append(tags_extractor(content, topk=20, w2v_model=w2v_model))
		if i==50: 
			for d in docs:
				print(d)
			break


	dictionary = corpora.Dictionary(docs)
	bows = [dictionary.doc2bow(doc) for doc in docs]
	tfidf_model = models.TfidfModel(bows)

	dictionary.save('../models/bow/1/corpora_dict.dict') # 重载用corpora.Dictionary.load(path)
	tfidf_model.save('../models/bow/1/corpora_tfidf.model') # 重载用models.TfidfModel.load(path)

	# 获取字典
	stoi = dictionary.token2id
	print("words num:",len(stoi))
	itos = dict(zip(stoi.values(), stoi.keys()))

	# test
	for i in range(20):
		test_doc = docs[i]
		test_bow = dictionary.doc2bow(test_doc)
		# 得到tf-idf表示
		test_tfidf = sorted(tfidf_model[test_bow], key=lambda x:x[1], reverse=True)
		print(test_doc)
		for item in test_tfidf[:5]:
			print(itos[item[0]], item[1])
		print()




if __name__ == '__main__':
	breakout_tracks = open("../data/breakout_tracks_set_1.txt").read().splitlines()
	no_breakout_tracks = open("../data/no_breakout_tracks_set_1.txt").read().splitlines()

	build_tfidf_model(breakout_tracks+no_breakout_tracks)

