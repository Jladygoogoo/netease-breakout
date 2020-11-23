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
    tags_pool = open("../data/metadata/自带tags.txt").read().splitlines()
    tags_d = {}
    for i, t in enumerate(tags):
        tags_d[t] = i

    vec = np.zeros(len(tags_pool))
    for t in tags:
        vec[tags_d[t]] = 1

    return vec


def get_reviews_df(filepath):
    with open(filepath) as f:
        content = json.load(f)

    data_corpus = []
    for item in content:
        data = []
        data.append((item['time']))
        data.append(item['content'].replace('\r',' '))
        data_corpus.append(data)

    df = pd.DataFrame(data_corpus, columns=["time", "content"])
    df.drop_duplicates(['time'], inplace=True)
    df['date'] = df['time'].map(lambda x:to_date(x))
    df.drop(['time'], axis=1, inplace=True)

    return df


def get_reviews_count(dates):
    '''
    params: filepath: .json评论文件
    return: reviews_count: 评论数目的时序数据
    return: dates_all: 和评论数目对应的日期字符串数组
    '''
    dates_counter = Counter(dates)
    first_date, last_date = sorted(dates_counter.keys())[0], \
                            sorted(dates_counter.keys())[-1]
    dates_all = get_every_day(first_date, last_date)

    dates_counter_all = {}
    for d in dates_all:
        if d in dates_counter:
            dates_counter_all[d] = dates_counter[d]
        else:
            dates_counter_all[d] = 0

    reviews_count = [x[1] for x in sorted(dates_counter_all.items(),key=lambda x:x[0])]
    return reviews_count, dates_all


def get_breakouts(sequence, window_size=10, thres=5, group=True, group_gap=15, min_reviews=100):
    '''
    params: sequence: 评论数的时序数组
    params: window_size: 考察窗口大小
    return: breakout_points = [(index, breakout_factor), ...]
    return: breakouts_group = [[(g1p1, bf11), (g1p2, bf12),...], [(g2p1, bf21),...], ...]
    '''
    breakout_factors = []
    k = window_size // 2
    for i in range(len(sequence)):
        start = max(0,i-k)
        end = min(i+k,len(sequence)-1)
        left = [sequence[i]-sequence[j] for j in range(start,i)]
        right = [sequence[i]-sequence[j] for j in range(i+1,end)]
        xi = np.sum(left+right)/(2*k)
        breakout_factors.append(xi)

    mean, std = np.mean(breakout_factors), np.std(breakout_factors)
    bf = np.array(list(filter(lambda x:(x-mean)/std, breakout_factors)))
    peaks_filter = np.where(bf>thres)

    # 根据breakout_factor筛选
    breakout_points = list(zip(np.array(range(len(sequence)))[peaks_filter], bf[peaks_filter]))
    # 根据min_reviews筛选
    breakout_points = [p for p in breakout_points if sequence[p[0]]>=min_reviews]

    if not group or len(breakout_points)==0:
        return breakout_points

    # 将临近的peaks合为一个group
    sorted_breakouts = list(sorted(breakout_points, key=lambda x:x[1],reverse=True))
    breakouts_group = [[sorted_breakouts[0]]]
    for p in sorted_breakouts[1:]:
        flag = 0
        for i in range(len(breakouts_group)):
            if abs(p[0]-breakouts_group[i][0][0]) < group_gap:
                breakouts_group[i].append(p)
                flag = 1
                break
        if not flag:
            breakouts_group.append([p])
                
    breakouts_group = sorted(breakouts_group,key=lambda l:l[0][0])

    return breakouts_group


def get_breakouts_text(df, date, max_reviews=2000):
    '''
    params: df: reviews_df结构
    params: date: %y-%m-%d，指定日期
    params: max_reviews，最大评论数据
    return: reviews_text，爆发评论文本
    '''
    reviews = df[df["date"]==date]["content"].values
    sample_size = min(max_reviews, len(reviews))
    reviews = random.sample(list(reviews), sample_size)

    return '\n'.join(reviews)
    





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




