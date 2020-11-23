import os
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
import functools
from collections import Counter
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import librosa
import sklearn.metrics as mt

from d2v import get_doc_vector
from preprocess import tags_extractor


def time_spent():
    def wrapper1(func):
        @functools.wraps(func)
        def wrapper2(*argv, **kwargs):
            start = time.time()
            result = func(*argv,**kwargs)
            end = time.time()
            
            m,s = divmod(end-start,60)
            h,m = divmod(m,60)
            print("time spent: {:02.0f}:{:02.0f}:{:02.0f}".format(h,m,s))
            return result
        return wrapper2
    return wrapper1

def std_timestamp(timestamp):
    timestamp = int (timestamp* (10 ** (10-len(str(timestamp)))))
    return timestamp

def to_date(timestamp):
    timestamp = int (timestamp* (10 ** (10-len(str(timestamp)))))
    dt = datetime.fromtimestamp(timestamp)
    date = datetime.strftime(dt,'%Y-%m-%d')
    return date


def get_every_day(begin_date, end_date, str_source=True):
    # 前闭后闭
    date_list = []
    if str_source:
        begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return date_list


# roc - 约登法寻找最佳阈值
def find_optimal_cutoff(tpr, fpr, thres):
    y = tpr - fpr
    youden_index = np.argmax(y)  # only the first occurrence is returned.
    optimal_thres = thres[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_thres, point


# 绘制roc曲线
def roc_auc_score(y_test, y_prob, plot=True):
    fpr, tpr, thres = mt.roc_curve(y_test, y_prob)
    roc_auc = mt.auc(fpr, tpr)
    optimal_thres, optimal_point = find_optimal_cutoff(tpr, fpr, thres)

    if plot:
        plt.plot(fpr, tpr, color='darkorange', 
            label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
        plt.text(optimal_point[0], optimal_point[1], "Threshold:{:.2f}".format(optimal_thres))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc, optimal_thres


def assign_dir(prefix, n_dir, dir_size, flag):
    '''
    指定各层文件夹文件数目，分配文件路径。
    params:
        + prefix
        + n_dir: 文件夹层数
        + dir_size: list, 各层文件夹的文件数目；int, 指定为所有层的大小
        + flag: 文件的编号（从0开始）
    return: 文件路径
    '''
    if type(dir_size)==int:
        dir_size = [dir_size]*n_dir
    if len(dir_size)!=n_dir:
        raise Exception("ERROR: dir_size tuple length is {} while n_dir is {}".format(len(dir_size), n_dir))
    
    dirs = []
    for i in range(n_dir-1, -1, -1):
        dirs.insert(0, str(flag % dir_size[i]))
        flag = flag // dir_size[i]
    dirs.insert(0, str(flag))

    dir_ = os.path.join(prefix, '/'.join(dirs[:-1]))
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    return dir_



def get_mfcc(path):
    '''
    param: path: 保存了提前提取的采样数据的.pkl文件路径
    return: mfcc序列 [<numpy>(20, n_frames)]
    '''
    with open(path, 'rb') as f:
        y = pickle.load(f)

    mfcc = librosa.feature.mfcc(y) # sr=22050
    return mfcc

def get_d2v_vector(r_path, model):
    '''
    param: path: 歌词.json文件路径
    param: model: doc2vec模型
    return: 文本向量 
    '''
    with open(r_path) as f:
        content = json.load(f)
    text = content["lrc"]["lyric"]
    if len(text)<2:
        return 0
    vec = get_doc_vector(text, model)

    return vec


def get_w2v_vector(word, model):
    if not model.wv.__contains__(word):
        return None
    return model.wv[word]


def get_tags_vector(tags):
    '''
    获取网易云内置tags的bags向量。
    '''
    tags_pool = open("../data/metadata/自带tags.txt").read().splitlines()
    tags_d = {}
    for i, t in enumerate(tags):
        tags_d[t] = i

    vec = np.zeros(len(tags_pool))
    for t in tags:
        vec[tags_d[t]] = 1

    return vec







if __name__ == '__main__':

    filepath = "/Volumes/nmusic/NetEase2020/data/proxied_reviews/0/0/3932174.json"
    reviews_count, dates =  get_reviews_count_series(filepath)
    # breakout_points = get_breakouts(reviews_count)
    # for p in breakout_points:
    #     print(dates[p[0]], p[1])
    breakouts_group = get_breakouts(reviews_count)
    for g in breakouts_group:
        print(g)
    x = range(len(dates))
    y = reviews_count
    plt.plot(x, y)
    for g in breakouts_group:
        if y[g[0][0]]>=50:
            plt.scatter(g[0][0], y[g[0][0]], c='r')

    plt.show()




