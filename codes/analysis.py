import os
import json
import pandas as pd
import numpy as np
import pickle
import datetime
from collections import Counter

import librosa
import matplotlib.pyplot as plt 
import seaborn as sns 
from pyecharts.charts import Bar
from pyecharts import options as opts
from sklearn.cluster import KMeans
from sklearn import metrics

from connect_db import MyConn
from utils import get_every_day
from breakout_tools import view_reviews_num_curve


def draw_hist(data, **kwargs):
    sns.displot(data, **kwargs)
    plt.show()


def draw_bar(data, render_path):
    print(data)
    bar = (
        Bar(init_opts={'width':'1200px','height':'2000px'})
        .add_xaxis(list(data.keys()))
        .add_yaxis(
            "爆发", 
            list(zip(*list(data.values())))[0],
            itemstyle_opts=opts.ItemStyleOpts(color='#499c9f')
        )
        .add_yaxis(
            "未爆发", 
            list(zip(*list(data.values())))[1],
            itemstyle_opts=opts.ItemStyleOpts(color='#FFA421')
        )
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position="right"))
        .set_global_opts(title_opts=opts.TitleOpts(title="自带标签统计"))
    )
    bar.render(render_path)



def basic_analysis(tracks_set):
    '''
    对指定的歌曲集进行基本分析：评论数、时间跨度...
    '''
    conn = MyConn()
    # 数据准备
    data = []
    targets = ["track_id", "tags", "reviews_num", "first_review", "last_review"]
    for tid in tracks_set:
        res = conn.query(targets=targets, conditions={"track_id": int(tid)})
        data.append(res[0])
    
    df = pd.DataFrame(data, columns=targets)
    # df.to_csv("../results/main_tagged_tracks/basic_info.csv", encoding="utf_8_sig", index=False)

    draw_hist(df["reviews_num"].values, log_scale=True, color="tab:orange")
    # durations = list(df.apply(lambda d: len(get_every_day(d["first_review"], d["last_review"], str_source=False)), axis=1).array)
    # draw_hist(durations)

    # tag

def mp3_analysis(tracks_set):
    '''
    对指定的歌曲集的音频时长进行分析。
    '''
    # 数据准备
    data_path = "../data/main_tagged_tracks/music_preload_data.pkl"
    with open(data_path,'rb') as f:
        data = pickle.load(f)

    durations = list(map(lambda x: librosa.get_duration(x[0],x[1]), data.values()))
    print("max duration:", np.max(durations))
    print("min duration:", np.min(durations))
    hist_view(durations)



def in_tags_analysis(breakouts_set, no_breakouts_set):
    '''
    对指定的歌曲集的内置tags情况进行分析。
    '''
    tags = open("../data/metadata/自带tags.txt").read().splitlines()
    breakouts_tags_d = {}
    no_breakouts_tags_d = {}
    for t in tags:
        breakouts_tags_d[t] = []
        no_breakouts_tags_d[t] = []

    conn = MyConn()
    for tid in breakouts_set:
        res = conn.query(targets=["tags"], conditions={"track_id":tid})[0]
        for t in res[0].split():
            breakouts_tags_d[t].append(tid)
    for tid in no_breakouts_set:
        res = conn.query(targets=["tags"], conditions={"track_id":tid})[0]
        for t in res[0].split():
            no_breakouts_tags_d[t].append(tid)

    tags_count = []
    for k in breakouts_tags_d:
        tags_count.append((k, (float(format(len(breakouts_tags_d[k])/1748*100,'.2f')), 
                                float(format(len(no_breakouts_tags_d[k])/10,'.2f')))))

    tags_count = sorted(tags_count, key=lambda x:x[1][0], reverse=False)
    draw_bar(dict(tags_count), "../data/main_tagged_tracks/tags_count.html")


def breakouts(json_path="../data/breakouts-u2.json"):
    '''
    基于json文件进行分析。
    '''
    # with open(json_path) as f:
    #   content = json.load(f)
    df = pd.read_json(json_path)
    # tracks = list(df["track_id"].unique())
    # print(df.shape)
    # print(len(tracks))
    # basic_analysis(tracks)
    values = df["reviews_num"].values
    print(np.median(values))
    draw_hist(list(filter(lambda x:x<=2000, values)), color="pink")
    # draw_hist(values)


def pos_neg_words():
    '''
    考察pairwise训练中的正负样本情况。
    '''
    path = "MyModel/models/pos_neg_words-3.pkl"
    with open(path, "rb") as f:
        content = pickle.load(f)
    for batch in content.values():
        neg_words = []
        for item in batch:
            neg_words.extend(item[1])
        print(Counter(neg_words).most_common(10))



def feature_words():
    '''
    考察 breakouts 和 no_breakouts 各自 feature_words 的特性
    '''
    def get_feature_words_counter(table):
        conn = MyConn()
        counter = Counter()
        res = [r[0].split() for r in conn.query(targets=["feature_words"], table=table)]
        for r in res:
          counter.update(r)
        return counter

    breakouts_counter = get_feature_words_counter("breakouts_feature_words_1")
    no_breakouts_counter = get_feature_words_counter("no_breakouts_feature_words_1")
    breakouts_input_size = sum(breakouts_counter.values())
    no_breakouts_input_size = sum(no_breakouts_counter.values())
    print(breakouts_input_size, no_breakouts_input_size)

    feature_words_set = set(list(breakouts_counter.keys()) + list(no_breakouts_counter.keys()))
    fw_freqency_d = {}
    for w in feature_words_set:
        b_freq = breakouts_counter[w]/breakouts_input_size if w in breakouts_counter else 0
        nb_freq = no_breakouts_counter[w]/no_breakouts_input_size if w in no_breakouts_counter else 0
        fw_freqency_d[w] = (b_freq, nb_freq)
    # print(fw_freqency_d)

    # 只保留词频较高的feature_words
    keep_rate = 0.0001
    fw_freqency_d = dict(filter(lambda p:p[1][0]>keep_rate or p[1][1]>keep_rate, fw_freqency_d.items()))
    freq_data = sorted([(k, (v[0]-v[1])*1000, v[0]*1000, v[1]*1000) for k,v in fw_freqency_d.items()], key=lambda p:abs(p[1]), reverse=True)
    for d in freq_data:
        print(d)

    df = pd.DataFrame(freq_data, columns=["feature_word", "diff", "breakout_freq", "no_breakout_freq"])
    df.to_csv("../records/feature_words_diff.csv", index=False, encoding="utf_8_sig")



def breakouts_curve():
    conn = MyConn()

    for i in range(6):
        save_dir = "../data/breakouts_curve_clusters/{}".format(i)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tracks = [r[0] for r in conn.query(targets=["track_id"], table="breakouts_complements", conditions={"label6":i})]
        for tid in tracks[:100]:
            save_path = os.path.join(save_dir, "{}.png".format(tid))
            view_reviews_num_curve(tid, save_path=save_path)


def remove_release_breakouts():
    conn = MyConn()
    breakouts = conn.query(targets=["id", "track_id", "date"], table="breakouts")

    release_breakouts_count = 0
    release_breakouts_tracks_set = set()
    more_breakouts_tracks_set = set()
    for b in breakouts:
        track_first_review = conn.query(targets=["first_review"], conditions={"track_id":b[1]}, fetchall=False)[0]
        if b[2] - track_first_review < datetime.timedelta(days=15):
            release_breakouts_count += 1
            release_breakouts_tracks_set.add(b[1])
            conn.update(table="breakouts", settings={"release_drive":1}, conditions={"id":b[0]})
        else:
            more_breakouts_tracks_set.add(b[1])

    print(release_breakouts_count)
    print(len(release_breakouts_tracks_set))
    print(len(more_breakouts_tracks_set))



def breakouts_complements():
    conn = MyConn()
    logspace = [(0,100), (100,180), (180,326), (326,589), (589,1066), (1066,3494), (3494,30000)]
    blevel_num = len(logspace)
    logspace_count = dict(zip(logspace, blevel_num*[0]))
    breakout_tracks = [r[0] for r in conn.query(targets=["DISTINCT(track_id)"], table="breakouts",
                                                conditions={"release_drive":0})]

    for track_id in breakout_tracks:
        reviews_num, first_review, last_review = conn.query(targets=["reviews_num","first_review","last_review"], 
                                    conditions={"track_id":track_id}, fetchall=False)
        breakouts = conn.query(targets=["flag","reviews_num","beta","release_drive"], table="breakouts", conditions={"track_id":track_id})
        days_num = (last_review - first_review).days
        # 除去爆发点的平均评论数
        avg_normal = float((reviews_num - np.sum([b[1] for b in breakouts])) / (days_num - len(breakouts)))

        blevel_vec = blevel_num*[0]
        for b in breakouts:
            if b[3]==1: continue # 不考虑release_drive爆发
            for i in range(blevel_num):
                if b[2]>=logspace[i][0] and b[2]<logspace[i][1]: # 考察beta区间
                    blevel_vec[i] += 1
                    logspace_count[logspace[i]] += 1
                    break

        breakouts_num = int(np.sum(blevel_vec))
        blevel = 0
        for i in range(len(blevel_vec)):
            blevel += i*blevel_vec[i]
        blevel = blevel*1.0 / breakouts_num
        settings = {
            "track_id": track_id,
            "average_reviews_num": avg_normal,
            "blevel_vec": ' '.join(map(str,blevel_vec)),
            "breakouts_num": breakouts_num,
            "blevel": blevel
        }
        conn.insert_or_update(table="breakouts_complements", settings=settings)
        # print(settings)
        print(track_id)


def breakouts_complements_cluster():
    '''
    利用breakouts_complements中average_reviews_num, breakouts_num, blevel / blevel_vec信息作聚类。
    测试多种聚类数目。
    并在breakouts_complements中更新track的聚类label。
    '''
    n_clusters = [5,6,7,8,9,10,11,12,13,14,15]
    conn = MyConn()
    res = conn.query(table="breakouts_complements", targets=["track_id","average_reviews_num","breakouts_num","blevel"])
    tracks = [r[0] for r in res]
    data = [r[1:] for r in res]

    # 使用blevel_vec
    # res = conn.query(table="breakouts_complements", targets=["track_id","average_reviews_num","blevel_vec"])
    # tracks = [r[0] for r in res]
    # data = []
    # for i in range(len(res)):
    #     tmp = [res[i][1]]
    #     tmp.extend(list(map(int, res[i][2].split())))
    #     data.append(tmp)

    # print(tracks[:2])
    # print(data[:2])

    def test_n_clusters():
        for n in n_clusters:
            model = KMeans(n_clusters=n, random_state=21)
            labels = model.fit_predict(data)
            n_labels = len(set(labels))
            print("n_clusters:", n)
            print("inertia:", model.inertia_)
            print("silhouette_score:", metrics.silhouette_score(data, labels))

            res = []
            for i in range(n_labels):
                child_data = [data[j] for j in range(len(data)) if labels[j]==i]
                # print("mean-average_reviews_num: {:.3f}, size: {}".format(np.mean(list(zip(*child_data))[0]), len(child_data)))
                res.append([np.mean(list(zip(*child_data))[0]), np.mean(list(zip(*child_data))[1]), np.mean(list(zip(*child_data))[2]), len(child_data)])
            res = sorted(res, key=lambda x:x[0])
            for r in res:
                print("mean-average_reviews_num: {:.3f}, mean-breakouts_num: {:.3f}, mean-blevel: {:.3f}, size: {}".format(
                    r[0], r[1], r[2], r[3]))
            print()

    test_n_clusters()
    # 指定n_clusters=6
    # model = KMeans(n_clusters=6, random_state=21)
    # labels = model.fit_predict(data)
    # for i in range(len(tracks)):
    #     track_id = tracks[i]
    #     label = labels[i]
    #     conn.update(table="breakouts_complements", settings={"label6":int(label)}, conditions={"track_id":track_id})
        # conn.update(table="breakouts_complements", settings={"label6_vec":int(label)}, conditions={"track_id":track_id})





if __name__ == '__main__':
    # tracks_set = pd.read_json("../data/breakouts-0.json")["track_id"].unique()
    # basic_analysis(tracks_set)
    # mp3_analysis(tracks_set)
    # in_tags_analysis(tracks_set, tracks_set2)
    # breakouts()
    # pos_neg_words()
    # feature_words()
    breakouts_curve()
    # remove_release_breakouts()
    # breakouts_complements()
    # breakouts_complements_cluster()
