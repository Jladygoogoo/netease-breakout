import os
import json
import pandas as pd
import numpy as np
import pickle
import traceback
import random
from threading import Thread, Lock
from multiprocessing import Lock as PLock
from queue import Queue
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
from gensim.models import Word2Vec
import soundfile as sf

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from breakout_tools import get_reviews_df, get_reviews_count, get_breakouts, get_breakouts_text, get_no_breakouts
from utils import assign_dir, get_tracks_set_db, get_dir_item_set, get_chorus
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
    #   f.write('\n'.join(map(str,u_set)))

    with open("../data/main_tagged_tracks/labels_dict.pkl", 'wb') as f:
        pickle.dump(main_tagged_tracks_d, f)


def extract_rawmusic(tracks_set, prefix, flag0=0):
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
                # 简单粗暴截取歌曲的10-40s内容
                # y, sr = librosa.load(mp3_path, duration=30, offset=10)
                # if librosa.get_duration(y, sr)<30:
                #     print("track-{} is shorter than 40s".format(track_id))
                #     continue

                # 检测并截取歌曲的副歌部分，设置时长为15s
                duration = 20
                # print(mp3_path)
                chorus = get_chorus(mp3_path, clip_length=duration)
                if chorus is None:
                    print("track-{}: failed to detect chorus.".format(track_id))
                    continue
                else:
                    chorus_start, chorus_end = chorus
                # y, sr = librosa.load(mp3_path, offset=start, duration=duration)
                y, sr = librosa.load(mp3_path)

                # 存储路径
                # pkl_path = os.path.join(
                #     assign_dir(prefix=task_args["prefix"], flag=task_args["flag"],
                #         n_dir=task_args["n_dir"], dir_size=task_args["dir_size"]),
                #     "{}.pkl".format(track_id)
                # )
                output_file = os.path.join(
                    assign_dir(prefix=task_args["prefix"], flag=task_args["flag"],
                        n_dir=task_args["n_dir"], dir_size=task_args["dir_size"]),
                    "{}.wav".format(track_id)
                )
                # 将flag+1以更新存储路径，需要lock
                task_args["lock"].acquire()
                task_args["flag"] += 1
                task_args["lock"].release()

                # if not os.path.exists(os.path.dirname(pkl_path)):
                #     os.makedirs(os.path.dirname(pkl_path))
                # with open(pkl_path, 'wb') as f:
                #     pickle.dump(y, f)
                # if not os.path.exists(os.path.dirname(pkl_path)):
                #     os.makedirs(os.path.dirname(pkl_path))
                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))

                chorus_wave_data = y[int((chorus_start-1)*sr) : int(chorus_end*sr)]
                sf.write(output_file, chorus_wave_data, sr)

                # break
                # print(thread_id, pkl_path)

            except KeyboardInterrupt:
                print("interrupted by keyboard.")
                break

            except:
                print("track-{}: failed to process.".format(track_id))
                print(traceback.format_exc())

    if input("flag={}，确定请输入yes: ".format(flag0)) != "yes":
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



def upload_chorus_start():
    def task(thread_id, task_args):
        conn = MyConn()
        while 1:
            sql = "SELECT track_id from tracks WHERE mp3_path IS NOT NULL AND chorus_start IS NULL"
            task_args["lock"].acquire()
            print("thread-{} acquire lock.".format(thread_id), end=" ")
            res = conn.query(sql=sql, fetchall=False)
            if res is not None:
                track_id = res[0]
                conn.update(settings={"chorus_start":0, "chorus_end":0}, conditions={"track_id":track_id})
            print("thread-{} release lock.".format(thread_id))
            task_args["lock"].release()
            if res is None: 
                break

            try:
                mp3_path = "/Volumes/nmusic/NetEase2020/data" + \
                    conn.query(targets=["mp3_path"], conditions={"track_id": track_id})[0][0]
                # 检测并截取歌曲的副歌部分，设置时长为20s
                duration = 20
                # print(mp3_path)
                chorus = get_chorus(mp3_path, clip_length=duration)
                if chorus is None:
                    print("track-{}: failed to detect chorus.".format(track_id))
                else:
                    chorus_start, chorus_end = chorus
                    conn.update(settings={"chorus_start":chorus_start, "chorus_end": chorus_end}, 
                            conditions={"track_id":track_id})
            except KeyboardInterrupt:
                conn.update(settings={"chorus_start":"null", "chorus_end": "null"}, 
                                            conditions={"track_id":track_id})
                break
            except:
                print(traceback.format_exc())

    # 开启thread群工作
    lock = Lock()
    task_args = {"lock":lock}
    threads_group = ThreadsGroup(task=task, n_thread=2, task_args=task_args)
    threads_group.start()






def upload_details():
    '''
    将歌曲的基本信息上传至数据库（歌曲名称、歌手姓名、专辑名称...）
    '''
    def extract_details(filename):
        with open(filename) as f:
            content = json.load(f)
        details = {
            "name": content["songs"][0]["name"],
            "artist": " ".join([item["name"] for item in content["songs"][0]["ar"]]),
            "pop": content["songs"][0]["pop"],
            "album": content["songs"][0]["al"]["name"]
        }
        return details

    read_path = "/Volumes/nmusic/NetEase2020/data/simple_proxied_tracks_details"
    conn = MyConn()


    for root, dirs, files in os.walk(read_path):
        for file in files:
            if "DS" in file: continue
            filepath = os.path.join(root, file)
            track_id = file[:-5]
            try:
                details = extract_details(filepath)
            except Exception as e:
                print(filepath)
                # print(traceback.format_exc())
                print(e)

            # print(details)
            conn.insert_or_update(table="details", settings={
                                "track_id": track_id, 
                                "name": details["name"],
                                "artist": details["artist"],
                                "album": details["album"],
                                "pop": details["pop"]})




def chorus_duration_distribution():
    conn = MyConn()
    sql = "SELECT chorus_start, chorus_end FROM tracks WHERE chorus_start IS NOT NULL AND chorus_end IS NOT NULL"
    res = conn.query(sql=sql)
    res = list(filter(lambda x:x[0]!=0, res))
    print(len(res))
    durations = [p[1]-p[0] for p in res]

    sns.displot(durations)
    plt.show()




'''
===============================
以下均是一些杂碎的不值得拥有姓名的操作
===============================
'''

def update_rawmusic_path():
    # conn = MyConn()
    # path = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if "DS" in file: continue
    #         track_id = file[:-4]
    #         pkl_path = os.path.join(root, file)
    #         try:
    #             conn.update(settings={"rawmusic_path":pkl_path}, conditions={"track_id":track_id})
    #         except:
    #             print("[ERROR]: {}".format(track_id))
    conn = MyConn()
    path = "/Volumes/nmusic/NetEase2020/data/proxied_lyrics"
    for root, dirs, files in os.walk(path):
        for file in files:
            if "DS" in file: continue
            track_id = file[:-5]
            lyrics_path = os.path.join(root, file)
            try:
                conn.update(table="tracks", settings={"lyrics_path":lyrics_path}, conditions={"track_id":track_id})
            except:
                print(traceback.format_exc())
                return
                print("[ERROR]: {}".format(track_id))


def build_train_test_dataset():
    conn = MyConn()
    random.seed(21)
    train_size, test_size = 3000, 1000
    size = train_size + test_size

    breakouts = random.sample([r[0] for r in conn.query(targets=["id"], conditions={"have_words": 1, "have_rawmusic":1}, table="breakouts")], size)
    breakouts_train, breakouts_test = breakouts[:train_size], breakouts[train_size:size+1]
    no_breakouts = random.sample([r[0] for r in conn.query(targets=["id"], conditions={"have_words": 1, "have_rawmusic":1}, table="no_breakouts")], size)
    no_breakouts_train, no_breakouts_test = no_breakouts[:train_size], no_breakouts[train_size:size+1]

    with open("../data/dataset/breakouts_id_train_2.txt", 'w') as f:
        f.write('\n'.join(breakouts_train))
    with open("../data/dataset/breakouts_id_test_2.txt", 'w') as f:
        f.write('\n'.join(breakouts_test))
    with open("../data/dataset/no_breakouts_id_train_2.txt", 'w') as f:
        f.write('\n'.join(no_breakouts_train))
    with open("../data/dataset/no_breakouts_id_test_2.txt", 'w') as f:
        f.write('\n'.join(no_breakouts_test))


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
    # path = "/Volumes/nmusic/NetEase2020/data/chorus_rawmusic"
    # conn = MyConn()
    # tracks_set = [r[0] for r in conn.query(targets=["track_id"], conditions={"have_mp3":1})]
    # tracks_set = random.sample(tracks_set, 300)
    # extract_rawmusic(tracks_set, prefix=path, flag0=300)
    # chorus_duration_distribution()
    # upload_chorus_start()

    # add_feature_words_to_db()
    # prep_no_breakouts_data()
    # build_train_test_dataset()
    upload_details()
