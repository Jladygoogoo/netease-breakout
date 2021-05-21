import os
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np 
import random
random.seed(21)
from collections import Counter, UserDict, OrderedDict

from gensim.models import Doc2Vec, Word2Vec
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data

from config import Config

sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from utils import get_mfcc, get_d2v_vector, get_w2v_vector, get_melspectrogram, get_reviews_vec, get_reviews_vec_with_freq, get_reviews_topk_words, get_track_filepath
from model_utils import get_mel_3seconds_groups
from connect_db import MyConn



class MyDataset(data.Dataset):
    '''
    构建数据集，定义 __getitem__ 方法。
    '''
    def __init__(self, track_label_pairs, config):
        self.config = config
        # self.conn = MyConn()
        self.data = track_label_pairs
        self.ids = list(range(len(self.data)))
        self.sub_tracks = pd.read_csv(config.SUB_TRACKS_CSV_PATH)

        # 相关模型与数据
        with open(config.ARTISTS_VEC_DICT_PATH, "rb") as f:
            self.d_artist_vec = pickle.load(f)   
        self.d2v_model = Doc2Vec.load(config.D2V_PATH)
        self.w2v_model = Word2Vec.load(config.W2V_PATH)
        with open(config.REVIEWS_FEATURE_WORDS_WITH_FREQS_POS_PATH) as f:
            self.d_breakouts_feature_words = json.load(f)
        with open(config.REVIEWS_FEATURE_WORDS_WITH_FREQS_NEG_PATH) as f:
            self.d_no_breakouts_feature_words = json.load(f)
        with open(config.D_POS_TRACK_BREAKOUT_PATH, 'rb') as f:
            self.d_pos_track_breakout = pickle.load(f)

        self.count_valid_pos = 0
        self.count_valid_neg = 0


    def __getitem__(self, index):
        '''
        必须定义，使用index获取一条完整的数据
        '''
        tid, label = self.data[index] # track_id和爆发标签（0/1）

        item_df = self.sub_tracks[self.sub_tracks["track_id"]==int(tid)]
        lyrics_path = get_track_filepath(tid, dir_path=self.config.LYRICS_DIR, file_fmt="json") # 对应歌词的文件位置
        artist = item_df[["artist"]].to_numpy()[0][0] # 对应艺人名称

        if self.config.MUSIC_DATATYPE=="vggish_examples":
            with open(get_track_filepath(tid, dir_path=self.config.VGGISH_EXAMPLES_DIR), "rb") as f:
                music_vec = pickle.load(f)            
        elif self.config.MUSIC_DATATYPE=="mel_3seconds_groups":
            with open(get_track_filepath(tid, dir_path=self.config.MEL_3SECONDS_GROUPS_DIR), "rb") as f:
                music_vec = pickle.load(f)
        lyrics_vec = torch.Tensor(get_d2v_vector(lyrics_path, self.d2v_model))
        artist_vec = torch.Tensor(self.d_artist_vec[artist.lower().strip()])
        reviews_vec = torch.Tensor(get_reviews_vec_with_freq(tid, breakout=label, w2v_model=self.w2v_model,
                                    d_breakouts=self.d_breakouts_feature_words, d_no_breakouts=self.d_no_breakouts_feature_words, 
                                    d_pos_track_breakout=self.d_pos_track_breakout, with_freq=False)[:self.config.TOPK])

        item_data = {
            "track_id": tid,
            "label": label,
            "music_vec": music_vec,
            "lyrics_vec": lyrics_vec,
            "artist_vec": artist_vec,
            "reviews_vec": reviews_vec,
        }

        return item_data

    def __len__(self):
        return len(self.ids)



def split_tracks_from_files(class_size, pos_tracks_filepath, neg_tracks_filepath, test_size=0.2):
    '''
    根据指定爆发和非爆发歌曲ID文件，构建 train-test 数据集。
    返回 (歌曲ID, label) 数组。
    '''
    pos_tracks = random.sample(open(pos_tracks_filepath).read().splitlines(), class_size)
    neg_tracks = random.sample(open(neg_tracks_filepath).read().splitlines(), class_size)
    pos_labels, neg_labels = [1]*class_size, [0]*class_size

    X_train, X_test, y_train, y_test = train_test_split(pos_tracks+neg_tracks, pos_labels+neg_labels, 
                                                test_size=test_size, random_state=21)
    train_data = list(zip(X_train, y_train)) # (track, label) pairs
    test_data = list(zip(X_test, y_test))

    return train_data, test_data



def get_data_loader(config):
    '''
    获取训练集和测试集的 dataloader。
    '''
    train_data, test_data = split_tracks_from_files(config.CLASS_SIZE,
                                    config.POS_TRACKS_FILEPATH, config.NEG_TRACKS_FILEPATH, config.TEST_SIZE)

    train_dataloader = data.DataLoader(MyDataset(train_data, config), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = data.DataLoader(MyDataset(test_data, config), batch_size=config.BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader



if __name__ == '__main__':
    config = Config()
        


