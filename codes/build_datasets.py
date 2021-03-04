import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd 
import logging
import traceback

from gensim.models import Doc2Vec

import torch

from connect_db import MyConn
from utils import get_mfcc, get_d2v_vector, get_tags_vector, roc_auc_score, time_spent
from preprocess import cut
from artists import generate_description_KB, generate_description_sup
# from MyModel.model import MusicFeatureExtractor, IntrinsicFeatureEmbed
# from MyModel.config import Config


def concatenate_features(features):
    feature_vec = np.concatenate(features, axis=None).ravel()
    return feature_vec


def get_X_y_embed(tracks_2_labels, conn, d2v_model, music_feature_extractor, intrinsic_feature_embed):
    # 构建.pkl文件的路径字典
    dir1 = "/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"
    dir2 = "/Volumes/nmusic/NetEase2020/data/no_breakouts_rawmusic"
    tid_2_mp3path_d = get_tid_2_mp3path_d(dir1)
    tid_2_mp3path_d2 = get_tid_2_mp3path_d(dir2)
    for tid in tid_2_mp3path_d2:
        if tid not in tid_2_mp3path_d:
            tid_2_mp3path_d[tid] = tid_2_mp3path_d2[tid]
    # print(len(tid_2_mp3path_d))

    X = []
    y = []
    flag = 1
    for tid, label in tracks_2_labels.items():
        tid = str(tid)
        try:
            # 获取音频特征向量
            mfcc = torch.tensor(get_mfcc(tid_2_mp3path_d[tid])).unsqueeze(0)
            # 获取歌词特征向量
            lyrics_path = conn.query(targets=["lyrics_path"],
                                            conditions={"track_id":tibd})[0][0]
            lyrics_vec = torch.tensor(get_d2v_vector(
                "/Volumes/nmusic/NetEase2020/data/"+lyrics_path, d2v_model))

            h1 = music_feature_extractor(mfcc).squeeze()
            # print(h1.shape)
            h2 = torch.cat((h1, lyrics_vec))
            # print(h2.shape)
            feature_vec = intrinsic_feature_embed(h2)

            X.append(feature_vec)
            y.append(label)

            print(flag, tid)
            flag += 1
        except KeyboardInterrupt:
            print("interrupted by user.")
            sys.exit(1)
        except:
            print("ERROR", tid)
            print(traceback.format_exc())
            # continue
            # break
    return X, y



def get_X(track_id, use_mp3, use_lyrics, use_artist, use_tags,
        lyrics_d2v_model, d_artist_vec):
    conn = MyConn()

    rawmusic_path, lyrics_path, artist = conn.query(
        table="sub_tracks", conditions={"track_id":track_id}, fetchall=False,
        targets=["rawmusic_path", "lyrics_path", "artist"])

    vecs = []
    if use_mp3:
        mfcc_vec = get_mfcc(rawmusic_path).ravel()
        vecs.append(mfcc_vec)
    if use_lyrics:
        lyrics_vec = get_d2v_vector(lyrics_path, lyrics_d2v_model)
        vecs.append(lyrics_vec)
    if use_artist:
        artist_vec = d_artist_vec[artist.lower().strip()]
        vecs.append(artist_vec)
    # if use_tags:
    #     tags_vec = get_tags_vector(tags.split())
    #     vecs.append(tags_vec)
    features_vec = concatenate_features(vecs)

    return features_vec



def build_dataset():
    conn = MyConn()
    dataset_size = 600
    pos_tracks = conn.query(sql="SELECT track_id FROM sub_tracks WHERE valid_bnum>0 AND rawmusic_path IS NOT NULL LIMIT {}".format(dataset_size))
    neg_tracks = conn.query(sql="SELECT track_id FROM sub_tracks WHERE valid_bnum=0 AND rawmusic_path IS NOT NULL LIMIT {}".format(dataset_size))
    lyrics_d2v_model = Doc2Vec.load("../models/d2v/d2v_a2.mod") # 歌词d2v模型
    with open("../data/artists_vec_dict.pkl", "rb") as f:
        d_artist_vec = pickle.load(f)

    X, y = [], []
    args = {
        "lyrics_d2v_model": lyrics_d2v_model,
        "d_artist_vec": d_artist_vec,
        "use_mp3": True,
        "use_lyrics": True,
        "use_artist": True,
        "use_tags": False
    }

    def add_data(tracks, label):
        for t in tracks:
            try:
                X.append(get_X(track_id=t, **args))
                y.append(label)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except:
                print(label, t)
                print(traceback.format_exc())
    add_data(pos_tracks, 1)
    add_data(neg_tracks, 0)

    dataset_name = "m"*args["use_mp3"] + "l"*args["use_lyrics"] + "a"*args["use_artist"] + "t"*args["use_tags"]\
                    + str(len(pos_tracks))
    with open("../data/dataset/{}.pkl".format(dataset_name), 'wb') as f:
        pickle.dump([X,y], f)




def build_dataset_less():
    filter_tracks = ["442314990", "5263408", "29418974", "742265"]
    ts1 = open("../data/main_tagged_tracks/tracks.txt").read().splitlines()[:1000]
    ts2 = open("../data/main_tagged_tracks/no_breakouts_tracks.txt").read().splitlines()

    # 是否爆发
    tracks_set = [(tid,1) for tid in ts1 if tid not in filter_tracks]
    tracks_set += [(tid,0) for tid in ts2 if tid not in filter_tracks]

    # 之后从mysql数据库中获取lyrics路径和tag
    conn = MyConn()
    d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

    X, y = get_X_y(dict(tracks_set), conn, d2v_model)

    with open("../data/main_tagged_tracks/dataset_violent_less.pkl", 'wb') as f:
        pickle.dump([X,y], f)



def build_dataset_embed(w_path):
    # ts1 = open("../data/main_tagged_tracks/tracks.txt").read().splitlines()[:1000]
    ts1 = list(pd.read_json("../data/breakouts-u2.json")["track_id"].unique())[:1000]
    ts2 = open("../data/no_breakouts_tracks.txt").read().splitlines()[:1000]
    print(len(ts1), len(ts2))

    # 是否爆发
    tracks_set = [(tid,1) for tid in ts1]
    tracks_set += [(tid,0) for tid in ts2]

    # 加载模型
    conn = MyConn()
    d2v_model = Doc2Vec.load("../models/d2v/d2v_a1.mod")

    config = Config()
    mf_path = "MyModel/models/3/mf_extractor-e3.pkl"
    if_path = "MyModel/models/3/if_embed-e3.pkl"
    music_feature_extractor = MusicFeatureExtractor(config)
    music_feature_extractor.load_state_dict(torch.load(mf_path))
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    intrinsic_feature_embed.load_state_dict(torch.load(if_path))
    music_feature_extractor.eval()
    intrinsic_feature_embed.eval()

    X, y = get_X_y_embed(dict(tracks_set), conn, d2v_model, music_feature_extractor, intrinsic_feature_embed)

    with open(w_path, 'wb') as f:
        pickle.dump([X,y], f)


if __name__ == '__main__':
    # w_path = "../data/mymodel_data/dataset_embed-3-1000.pkl"
    build_dataset()


