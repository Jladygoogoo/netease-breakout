import os
import sys
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
from utils import count_files, get_chorus


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



def extract_chorus_mark_rawmusic():
    '''
    基于每首歌的chorus_start和chorus_end提取rawmusic。
    使用多线程。
    '''
    conn = MyConn()
    sql = "SELECT track_id FROM sub_tracks WHERE rawmusic_path IS NULL AND valid_bnum=0 AND chorus_start IS NOT NULL AND chorus_start>0 LIMIT 1100"
    tracks = [r[0] for r in conn.query(sql=sql)]
    
    save_dir_prefix = "/Volumes/nmusic/NetEase2020/data/chorus_mark_rawmusic"
    n_dir, dir_size = 1, 100
    flag, saved_files = count_files(save_dir_prefix, return_files=True) # 已经提取好的歌曲
    saved_tracks = [x[:-4] for x in saved_files]

    q_tracks = Queue()
    for t in tracks: 
        if t not in saved_tracks:
            q_tracks.put(t)
    print(q_tracks.qsize())
    lock = Lock()

    def task(thread_id, task_args):
        conn = MyConn()
        while not q_tracks.empty():
            try:
                tid = q_tracks.get()
                
                lock.acquire()
                dirpath = assign_dir(prefix=save_dir_prefix, flag=task_args["flag"],
                                     n_dir=n_dir, dir_size=dir_size)
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                filepath = os.path.join(dirpath, "{}.pkl".format(tid))
                task_args["flag"] += 1
                lock.release()

                # 从数据库中获取歌曲的 chorus_start, mp3_path
                chorus_start, mp3_path = conn.query(
                    table="sub_tracks", targets=["chorus_start", "mp3_path"], 
                    conditions={"track_id":tid}, fetchall=False)

                y, sr = librosa.load(mp3_path, offset=chorus_start, duration=20) # 副歌识别设定为20s

                with open(filepath, 'wb') as f:
                    pickle.dump(y, f)
            except KeyboardInterrupt:
                print("KeyboardInterrupt. q_tracks size: {}".format(q_tracks.qsize()))
                break
            except:
                print(tid)
                print(traceback.format_exc())
        sys.exit(0)

    task_args = {}
    task_args["flag"] = flag
    threads_group = ThreadsGroup(task=task, n_thread=10, task_args=task_args)
    threads_group.start()



    

def chorus_duration_distribution():
    conn = MyConn()
    sql = "SELECT chorus_start, chorus_end FROM tracks WHERE chorus_start IS NOT NULL A　ND chorus_end IS NOT NULL"
    res = conn.query(sql=sql)
    res = list(filter(lambda x:x[0]!=0, res))
    print(len(res))
    durations = [p[1]-p[0] for p in res]

    sns.displot(durations)
    plt.show()




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
    extract_chorus_mark_rawmusic()



