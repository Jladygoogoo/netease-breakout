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

import torch
import librosa
from gensim.models import Word2Vec
import soundfile as sf

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from breakout_tools import get_reviews_df, get_reviews_count, get_breakouts, get_breakouts_text, get_no_breakouts
from utils import assign_dir, get_tracks_set_db, get_dir_item_set, get_chorus, get_mel_3seconds_groups
from preprocess import tags_extractor
from utils import count_files, get_chorus
from vggish_input import waveform_to_examples
from config import Config


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
    sql = "SELECT track_id FROM sub_tracks WHERE rawmusic_path IS NULL AND valid_bnum=0 AND chorus_start>0"
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



def build_vggish_embed_dataset():
    conn = MyConn()
    sql = "SELECT track_id FROM sub_tracks WHERE is_valid=1 AND vggish_embed_path IS NULL"
    tracks = [r[0] for r in conn.query(sql=sql)]
    
    save_dir_prefix = "/Volumes/nmusic/NetEase2020/data/vggish_embed"
    n_dir, dir_size = 1, 100
    flag, saved_files = count_files(save_dir_prefix, return_files=True) # 已经提取好的歌曲
    saved_tracks = [x[:-4] for x in saved_files]

    q_tracks = Queue()
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=True)
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
                rawmusic_path = conn.query(
                    table="sub_tracks", targets=["rawmusic_path"], 
                    conditions={"track_id":tid}, fetchall=False)[0]

                with open(rawmusic_path, "rb") as f:
                    y = pickle.load(f)
                embed = vggish(y, fs=22050)

                with open(filepath, 'wb') as f:
                    pickle.dump(embed, f)
            except KeyboardInterrupt:
                print("KeyboardInterrupt. q_tracks size: {}".format(q_tracks.qsize()))
                break
            except:
                print(tid)
                print(traceback.format_exc())
        sys.exit(0)

    task_args = {}
    task_args["flag"] = flag
    threads_group = ThreadsGroup(task=task, n_thread=5, task_args=task_args)
    threads_group.start()
    


def build_vggish_examples_dataset():
    conn = MyConn()
    tracks = [r[0] for r in conn.query(table="sub_tracks", targets=["track_id"], conditions={"is_valid":1})]
    save_dir_prefix = "/Volumes/nmusic/NetEase2020/data/vggish_examples"
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
                rawmusic_path = conn.query(
                    table="sub_tracks", targets=["rawmusic_path"], 
                    conditions={"track_id":tid}, fetchall=False)[0]

                with open(rawmusic_path, "rb") as f:
                    y = pickle.load(f)
                    music_vec = waveform_to_examples(y, sample_rate=22050).squeeze(1)

                with open(filepath, 'wb') as f:
                    pickle.dump(music_vec, f)
            except KeyboardInterrupt:
                print("KeyboardInterrupt. q_tracks size: {}".format(q_tracks.qsize()))
                break
            except:
                print(tid)
                print(traceback.format_exc())
        sys.exit(0)

    task_args = {}
    task_args["flag"] = flag
    threads_group = ThreadsGroup(task=task, n_thread=5, task_args=task_args)
    threads_group.start()


def build_mel_3seconds_groups_dataset():
    conn = MyConn()
    config = Config()
    tracks = [r[0] for r in conn.query(table="sub_tracks", targets=["track_id"], conditions={"is_valid":1})]
    save_dir_prefix = "/Volumes/nmusic/NetEase2020/data/mel_3seconds_groups"
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
                mp3_path, chorus_start = conn.query(
                    table="sub_tracks", targets=["mp3_path", "chorus_start"], 
                    conditions={"track_id":tid}, fetchall=False)

                music_vec = get_mel_3seconds_groups(mp3_path, config, offset=chorus_start, duration=18)

                with open(filepath, 'wb') as f:
                    pickle.dump(music_vec, f)
            except KeyboardInterrupt:
                print("KeyboardInterrupt. q_tracks size: {}".format(q_tracks.qsize()))
                break
            except:
                print(tid)
                print(traceback.format_exc())
        sys.exit(0)

    task_args = {}
    task_args["flag"] = flag
    threads_group = ThreadsGroup(task=task, n_thread=5, task_args=task_args)
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


def build_valid_tracks():
    '''
    生成可用的数据集
    '''
    # pos
    conn = MyConn()
    valid_breakouts = [r[0] for r in conn.query(sql="SELECT id FROM breakouts WHERE is_valid=1 and simi_score>=0.5 and reviews_num>=100 and beta>=50")]
    pos_r_path = "../data/reviews_feature_words_with_freqs/breakouts_cls.json"
    neg_r_path = "../data/reviews_feature_words_with_freqs/no_breakouts_cls.json"
    pos_w_path = "../data_related/tracks/pos_tracks_cls_vgg.txt"
    pos_d_w_path = "../data_related/tracks/d_pos_track_breakout_cls_vgg.pkl"
    neg_w_path = "../data_related/tracks/neg_tracks_cls_vgg.txt"

    # pos
    with open(pos_r_path) as f:
        data = json.load(f)
        breakouts_with_music_words = [bid for bid in data if data[bid]["len"]>=5]
    valid_breakouts = [bid for bid in valid_breakouts if bid in breakouts_with_music_words]
    d_pos_track_breakout = {}
    for bid in valid_breakouts:
        tid = bid.split('-')[0]
        if tid not in d_pos_track_breakout:
            d_pos_track_breakout[tid] = bid
    print("d size:", len(d_pos_track_breakout))
    valid_pos_tracks = [r[0] for r in conn.query(sql="SELECT track_id FROM sub_tracks WHERE language in ('ch','en') and vggish_examples_path is not null and valid_bnum>0")]
    pos_tracks_with_valid_breakouts = list(set([bid.split('-')[0] for bid in valid_breakouts]))
    valid_pos_tracks = [tid for tid in valid_pos_tracks if tid in pos_tracks_with_valid_breakouts]
    print(len(valid_pos_tracks))
    with open(pos_w_path, 'w') as f:
        f.write("\n".join(valid_pos_tracks))
    with open(pos_d_w_path, "wb") as f:
        pickle.dump(d_pos_track_breakout, f)

    # neg
    with open(neg_r_path) as f:
        data = json.load(f)
        neg_tracks_with_music_words = [tid for tid in data if data[tid]["len"]>=5]
    valid_neg_tracks = [r[0] for r in conn.query(sql="SELECT track_id FROM sub_tracks WHERE language in ('ch','en') and vggish_examples_path is not null and valid_bnum=0")]
    valid_neg_tracks = [tid for tid in valid_neg_tracks if tid in neg_tracks_with_music_words]
    print(len(valid_neg_tracks))
    with open(neg_w_path, 'w') as f:
        f.write("\n".join(valid_neg_tracks))



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
    # build_vggish_embed_dataset()
    # extract_chorus_mark_rawmusic()
    # build_valid_tracks()
    # build_vggish_examples_dataset()
    build_mel_3seconds_groups_dataset()



