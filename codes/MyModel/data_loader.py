import os
import os
import sys
import json
import pickle
import numpy as np 
import random
random.seed(21)
from collections import Counter, UserDict

from gensim.models import Doc2Vec, Word2Vec

import torch
import torch.utils.data as data

from config import Config

sys.path.append("/Users/inkding/Desktop/netease2/codes")
from utils import get_mfcc, get_d2v_vector, get_w2v_vector
from connect_db import MyConn


class MyDataset(data.Dataset):
    '''
    + music feature[np.ndarray ~ (20, -1)]: mfcc特征
    + lyrics feature[np.ndarray ~ (300,)]: 歌词特征
    + artist feature[np.ndarray ~ (72,)]: 歌手特征
    + reviews feature[np.ndarray ~ (5, 300)]: 评论特征
    '''
    def __init__(self, config, mode):
        self.config = config
        self.conn = MyConn()

        # 获取正负样本集
        if mode=="train":
            DATASET_SIZE = config.TRAIN_DATASET_SIZE
            DATASET_OFFSET = 0
        else:
            DATASET_SIZE = config.TEST_DATASET_SIZE
            DATASET_OFFSET = config.TRAIN_DATASET_SIZE//2
        sql = "SELECT track_id FROM sub_tracks WHERE valid_bnum{}0 AND rawmusic_path IS NOT NULL LIMIT {},{}"
        b_tracks = self.conn.query(sql=sql.format('>', DATASET_OFFSET, DATASET_SIZE//2))
        nb_tracks = self.conn.query(sql=sql.format('=', DATASET_OFFSET, DATASET_SIZE//2))
        self.ps_track_label = [(tid, 1) for tid in b_tracks] + [(tid, 0) for tid in nb_tracks]
        self.ids = list(range(DATASET_SIZE))

        # 相关模型与数据
        with open(config.ARTISTS_VEC_DICT_PATH, "rb") as f:
            self.d_artist_vec = pickle.load(f)
        self.d2v_model = Doc2Vec.load(config.D2V_PATH)
        self.w2v_model = Word2Vec.load(config.W2V_PATH)


    def __getitem__(self, index):
        '''
        必须定义，使用index获取一条完整的数据
        '''
        tid, label = self.ps_track_label[index]
        rawmusic_path, lyrics_path, artist = self.conn.query(
            table="sub_tracks", conditions={"track_id":tid}, fetchall=False,
            targets=["rawmusic_path", "lyrics_path", "artist"])

        artist_vec, reviews_vec = None, None # artist和reviews信息是否使用
        mfcc_vec = torch.Tensor(get_mfcc(rawmusic_path))
        lyrics_vec = torch.Tensor(get_d2v_vector(lyrics_path, self.d2v_model))

        item_data = {
            "track_id": tid,
            "label": label,
            "mfcc_vec": mfcc_vec,
            "lyrics_vec": lyrics_vec
        }

        if self.config.USE_ARTIST:
            artist_vec = self.d_artist_vec[artist.lower().strip()]
            item_data["artist_vec"] = torch.Tensor(artist_vec)
        if self.config.USE_REVIEWS:
            item_data["reviews_vec"] = torch.Tensor(reviews_vec)

        return item_data


    def __len__(self):
        return len(self.ids)


def get_data_loader(config, mode, batch_size):
    my_dataset = MyDataset(config, mode)
    data_loader = data.DataLoader(dataset=my_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    return data_loader



if __name__ == '__main__':
    config = Config()
    data_loader = get_data_loader(config)

    for i, item_data in enumerate(data_loader):
        for v in item_data.values():
            print(v)


