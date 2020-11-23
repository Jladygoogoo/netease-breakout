import os
import sys
import pickle
import time
import numpy as np
import pandas as pd 
import logging
import traceback

from gensim.models import Doc2Vec

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as mt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import torch

from connect_db import MyConn
from utils import get_mfcc, get_d2v_vector, get_tags_vector, roc_auc_score, time_spent
from MyModel.model import MusicFeatureExtractor, IntrinsicFeatureEmbed
from MyModel.config import Config
# import build_datasets 



# 二分类模型比较
def compare_models():
	# random forest

	def basic_eval(y_test, y_pred, y_prob):
		print("accuracy: {:.3f}".format(mt.precision_score(y_test, y_pred)))
		print("f1-score: {:.3f}".format(mt.f1_score(y_test, y_pred)))

		roc_auc, best_thres = roc_auc_score(y_test, y_prob, plot=False)
		print("roc_auc: {:.3f} (best_thres={:.3f})".format(roc_auc, best_thres))


	# random forest
	@time_spent()
	def rf(X, y):
		max_depth = 10
		n_estimators = 30
		model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=21)
		X_train, X_test, y_train, y_test = train_test_split(
										X, y, test_size=.3, random_state=21)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		y_prob = model.predict_proba(X_test)[:,1]
		
		print(model.__class__.__name__)
		print("params: max_depth: {}, n_estimators: {}".format(max_depth, n_estimators))
		basic_eval(y_test, y_pred, y_prob)


	# svc
	@time_spent()
	def svc(X, y):
		gamma = 0.1
		C = 1
		model = SVC(gamma=gamma, C=C, random_state=21)

		X_train, X_test, y_train, y_test = train_test_split(
										X, y, test_size=.3, random_state=21)

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		y_prob = model.decision_function(X_test)

		print("SVC")
		print("params: gamma: {}, C: {}".format(gamma, C))
		basic_eval(y_test, y_pred, y_prob)

	@time_spent()
	def adaboost(X, y):
		learning_rate = 0.1
		n_estimators = 50
		model = AdaBoostClassifier(learning_rate=learning_rate,
								n_estimators=n_estimators, random_state=21)

		X_train, X_test, y_train, y_test = train_test_split(
										X, y, test_size=.3, random_state=21)

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		y_prob = model.predict_proba(X_test)[:,1]

		print("AdaBoost")
		print("params: learning_rate: {}, n_estimators: {}".format(learning_rate, n_estimators))
		basic_eval(y_test, y_pred, y_prob)


	# lightGBM
	@time_spent()
	def lgbm(X, y):
		# 这里必须转换为numpy.ndarray的形式，否则会报错
		X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=.3, random_state=21)
		train_data = lgb.Dataset(X_train, label=y_train)
		test_data = lgb.Dataset(X_test, label=y_test)
		params={
		    'learning_rate':0.1,
		    'lambda_l1':0.1,
		    'lambda_l2':0.2,
		    'max_depth':6,
		    "num_leaves": 50,
		    'objective':'binary',
		}
		model = lgb.train(params, train_data, valid_sets=[test_data])
		# 注意lgb的predict相当于sklearn中的predict_proba
		y_prob = model.predict(X_test) # n_dim=1，正例的概率
		# 取 thres = 0.5
		y_pred = np.where(y_prob>=0.5, 1, 0)
		print("LightGBM")
		print(params)
		basic_eval(y_test, y_pred, y_prob)
		
		

	# 载入数据
	# with open("../data/main_tagged_tracks/dataset_violent_less.pkl", 'rb') as f:
	with open("../data/mymodel_data/dataset_embed-3-1000.pkl", 'rb') as f:
		X,y = pickle.load(f)

	X = list(map(lambda x:x.detach().numpy(), X))
	# 标准化
	# X = StandardScaler().fit_transform(X)
	# print(y)
	# rf(X, y)
	# svc(X, y)
	# adaboost(X, y)
	lgbm(X, y)


# 多分类模型比较
def compare_models_multiclass():
	def basic_eval(model, y_test, y_pred, y_prob):
		# 混淆矩阵
		print(mt.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

		# accuracy, precision, recall, f1
		print("precision: {:.3f}".format(mt.precision_score(y_test, y_pred, average="micro")))
		print("recall: {:.3f}".format(mt.recall_score(y_test, y_pred, average="micro")))
		print("f1-score: {:.3f}".format(mt.f1_score(y_test, y_pred, average="micro")))
		# if y_prob != None:
		roc_auc = mt.roc_auc_score(y_test, y_prob, average="micro")
		print("roc_auc: {:.3f}".format(roc_auc))

		# 详细报告
		# print(mt.classification_report(y_test, y_pred, digits=4))


	# 模型1: 使用 OneVsRest + SVC
	@time_spent()
	def ovr_svc(X, y):
		# 需要将y转换为one-hot形式
		y = LabelBinarizer().fit_transform(y)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

		model = OneVsRestClassifier(LinearSVC(random_state=21, verbose=1))
		model.fit(X_train, y_train)

		# 计算micro类型的AUC
		y_prob = model.decision_function(X_test)
		y_pred = model.predict(X_test)

		basic_eval(model, y_test, y_pred, y_prob)


	# 模型2: 直接使用 LogisticRegression
	@time_spent()
	def lr(X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
		# 注意此处将 multi_class 设置为 "ovr"
		# model = LogisticRegression(solver='sag',multi_class='ovr', verbose=1) 
		model = LogisticRegression(solver='sag',multi_class='ovr') 
		model.fit(X_train, y_train)

		y_prob = model.predict_proba(X_test)
		y_pred = model.predict(X_test)

		bi_y_test = label_binarize(y_test, classes=[0,1,2,3,4])
		bi_y_pred = label_binarize(y_pred, classes=[0,1,2,3,4])

		basic_eval(model, bi_y_test, bi_y_pred, y_prob)


	# 模型3: 使用 RandomForest
	@time_spent()
	def rf(X, y):
		# 需要将y转换为one-hot形式
		max_depth = 10
		n_estimators = 50
		model = RandomForestClassifier()

		y = LabelBinarizer().fit_transform(y)
		X_train, X_test, y_train, y_test = train_test_split(
									X, y, test_size=.3, random_state=21)
		model.fit(X_train, y_train)

		# 计算micro类型的AUC
		y_pred = model.predict(X_test)
		basic_eval(model, y_test, y_pred, None)		



	# 模型4: 使用集成树 lightgbm
	@time_spent()
	def lgbm(X, y):
		# 这里必须转换为numpy.ndarray的形式，否则会报错
		X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=.3)
		train_data = lgb.Dataset(X_train, label=y_train)
		test_data = lgb.Dataset(X_test, label=y_test)
		params={
		    'learning_rate':0.1,
		    'lambda_l1':0.1,
		    'lambda_l2':0.2,
		    'max_depth':6,
		    'objective':'multiclass',
		    'num_class':5,
		}
		model = lgb.train(params, train_data, valid_sets=[test_data])
		# 注意lgb的predict相当于sklearn中的predict_proba
		y_prob = model.predict(X_test) 
		y_pred = y_prob.argmax(axis=1)

		bi_y_test = label_binarize(y_test, classes=[0,1,2,3,4])
		bi_y_pred = label_binarize(y_pred, classes=[0,1,2,3,4])

		basic_eval(model, bi_y_test, bi_y_pred, y_prob)

	# 正式开始
	with open("../data/main_tagged_tracks/dataset_violent_multiclass.pkl", 'rb') as f:
		X,y = pickle.load(f)
	X = StandardScaler().fit_transform(X)
	# print(X.shape)
	# print(y)
	# ovr_svc(X, y)
	# lr(X, y)
	# rf(X, y)
	lgbm(X, y)

def compare_models_multilabels():

	with open("../data/main_tagged_tracks/dataset_violent_multilabels.pkl", 'rb') as f:
		X,y = pickle.load(f)
	X = StandardScaler().fit_transform(X)



if __name__ == '__main__':
	# build_dataset()
	# build_dataset_multiclass()
	# build_dataset_less()
	compare_models()
	# compare_models_multiclass()
	# compare_models_multilabels()
	# build_dataset_embed()



 