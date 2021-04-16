import os
import os
import sys
import json
import pickle
import numpy as np 
import random
random.seed(21)
from collections import Counter, UserDict, OrderedDict

from gensim.models import Doc2Vec, Word2Vec

import torch
import torch.utils.data as data

from config import Config

sys.path.append("/Users/inkding/Desktop/netease2/codes")
from utils import get_mfcc, get_d2v_vector, get_w2v_vector, get_melspectrogram, get_reviews_vec, get_reviews_vec_with_freq, get_reviews_topk_words
from model_utils import get_mel_3seconds_groups
from connect_db import MyConn



class MyDataset(data.Dataset):
    '''
    构建数据集，定义 __getitem__ 方法。
    '''
    def __init__(self, track_label_pairs, config):
        self.config = config
        self.conn = MyConn()
        # self.vggish = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=True)

        self.data = track_label_pairs
        self.ids = list(range(len(self.data)))

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
        tid, label = self.data[index]
        rawmusic_path, vggish_embed_path, vggish_examples_path, mel_3seconds_groups_path, lyrics_path, artist, chorus_start = self.conn.query(
            table="sub_tracks", conditions={"track_id":tid}, fetchall=False,
            targets=["rawmusic_path", "vggish_embed_path", "vggish_examples_path", "mel_3seconds_groups_path", "lyrics_path", "artist", "chorus_start"])

        if self.config.MUSIC_DATATYPE=="vggish":
            with open(vggish_embed_path, "rb") as f:
                music_vec = pickle.load(f)
        elif self.config.MUSIC_DATATYPE=="mel":
            music_vec = torch.Tensor(get_melspectrogram(rawmusic_path, config=self.config))
        elif self.config.MUSIC_DATATYPE=="mfcc":
            music_vec = torch.Tensor(get_mfcc(rawmusic_path, config=self.config))
        elif self.config.MUSIC_DATATYPE=="vggish_examples":
            with open(vggish_examples_path, "rb") as f:
                music_vec = pickle.load(f)            
        elif self.config.MUSIC_DATATYPE=="mel_3seconds_groups":
            with open(mel_3seconds_groups_path, "rb") as f:
                music_vec = pickle.load(f)
        lyrics_vec = torch.Tensor(get_d2v_vector(lyrics_path, self.d2v_model))
        artist_vec = torch.Tensor(self.d_artist_vec[artist.lower().strip()])
        # reviews_vec = torch.Tensor(get_reviews_vec(tid, breakout=label, 
        #             w2v_model=self.w2v_model, key=self.config.REVIEWS_VEC_KEY)[:self.config.TOPK])
        # reviews_topk_words = get_reviews_topk_words(tid, is_breakout=label, key="candidates")[:5]
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
            # "reviews_topk_words": reviews_topk_words
        }

        return item_data

    def __len__(self):
        return len(self.ids)



def split_tracks_from_db(train_size, valid_size, test_size, random_state=21):
    '''
    从数据库中获取可用歌曲id，并将其划分为 train, valid, test
    '''
    total_size = train_size + valid_size + test_size
    class_train_size, class_valid_size, class_test_size = train_size//2, valid_size//2, test_size//2
    conn = MyConn()
    pos_sql = "SELECT track_id FROM sub_tracks WHERE valid_bnum>0 AND is_valid=1"
    neg_sql = "SELECT track_id FROM sub_tracks WHERE valid_bnum=0 AND is_valid=1"
    pos_tracks = random.sample([(r[0], 1) for r in conn.query(sql=pos_sql)], total_size//2)
    neg_tracks = random.sample([(r[0], 0) for r in conn.query(sql=neg_sql)], total_size//2)
    if len(pos_tracks)<total_size//2:
        raise RuntimeError("dataset size too large.")

    train_tracks = pos_tracks[:class_train_size] + neg_tracks[:class_train_size]
    valid_tracks = pos_tracks[class_train_size:class_train_size+class_valid_size] + neg_tracks[class_train_size:class_train_size+class_valid_size]
    test_tracks = pos_tracks[-class_test_size:] + neg_tracks[-class_test_size:]

    return train_tracks, valid_tracks, test_tracks


def split_tracks_from_files(train_size, valid_size, test_size, 
            pos_tracks_filepath, neg_tracks_filepath, random_state=21):
    total_size = train_size + valid_size + test_size
    class_train_size, class_valid_size, class_test_size = train_size//2, valid_size//2, test_size//2
    
    pos_tracks = [(r, 1) for r in open(pos_tracks_filepath).read().splitlines()]
    neg_tracks = [(r, 0) for r in open(neg_tracks_filepath).read().splitlines()]

    train_tracks = pos_tracks[:class_train_size] + neg_tracks[:class_train_size]
    valid_tracks = pos_tracks[class_train_size:class_train_size+class_valid_size] + neg_tracks[class_train_size:class_train_size+class_valid_size]
    test_tracks = pos_tracks[-class_test_size:] + neg_tracks[-class_test_size:]

    return train_tracks, valid_tracks, test_tracks


def get_data_loader(config):
    '''
    获取 data_loaders
    '''
    # train_tracks, valid_tracks, test_tracks = split_tracks_from_db(
    #                     config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE)
    # train_dataloader = data.DataLoader(MyDataset(train_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)
    # valid_dataloader = data.DataLoader(MyDataset(valid_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)
    # test_dataloader = data.DataLoader(MyDataset(test_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)

    train_tracks, valid_tracks, test_tracks = split_tracks_from_files(
            config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE,
            config.POS_TRACKS_FILEPATH, config.NEG_TRACKS_FILEPATH)
    train_dataloader = data.DataLoader(MyDataset(train_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)
    valid_dataloader = data.DataLoader(MyDataset(valid_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = data.DataLoader(MyDataset(valid_tracks, config), batch_size=config.BATCH_SIZE, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader



if __name__ == '__main__':
    config = Config()
    config.MUSIC_DATATYPE="vggish"
    config.REVIEWS_VEC_KEY = "candidates"
    train_dataloader, valid_dataloader, test_dataloader = get_data_loader(config)

    counter_pos = Counter()
    counter_neg = Counter()
    for batch_data in train_dataloader:
        labels = batch_data["label"]
        words = batch_data["reviews_topk_words"]
        for i in range(len(words[0])):
            item_words = [words[k][i] for k in range(5)]
            if labels[i]==1:
                counter_pos.update(item_words)
            else:
                counter_neg.update(item_words)
    total_pos, total_neg = sum(counter_pos.values()), sum(counter_neg.values())
    d_freq_pos = dict([(p[0],p[1]/total_pos) for p in counter_pos.most_common()])
    d_freq_neg = dict([(p[0],p[1]/total_neg) for p in counter_neg.most_common()])

    d_freq_diff = {}
    for w in d_freq_pos:
        if w in d_freq_neg:
            d_freq_diff[w] = d_freq_pos[w] / d_freq_neg[w]
        else:
            d_freq_diff[w] = -1
    for w in d_freq_pos:
        if w not in d_freq_diff:
            d_freq_diff[w] = -1

    d_freq_diff_sorted = OrderedDict(sorted(list(d_freq_diff.items()), key=lambda p:p[1], reverse=True))
    for k,v in d_freq_diff_sorted.items():
        print(k, v)


