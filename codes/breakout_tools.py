import os
import random
import pandas as pd 
import numpy as np
from collections import Counter

from utils import get_every_day



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
    
