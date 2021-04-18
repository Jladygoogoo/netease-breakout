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
from gensim.models import Word2Vec
# from pychorus.helpers import create_chroma, find_chorus


from d2v import get_doc_vector
from preprocess import tags_extractor
from connect_db import MyConn


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)

    return similiarity


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
    '''
    将网易云评论的timestamp变为日期
    '''
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
    '''
    指定各层文件夹文件数目，分配文件路径。
    params:
        + prefix
        + n_dir: 文件夹层数
        + dir_size: list, 各层文件夹的文件数目；int, 指定为所有层的大小
        + flag: 文件的编号（从0开始）
    return: 文件路径
    '''
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


def count_files(dirpath, return_files=False):
    '''
    统计文件夹下的文件数量
    如果 return_files=True，则返回所有的文件名称
    '''
    count = 0
    saved_files = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if "DS" in file: continue
            saved_files.append(file)
            count += 1
    if return_files:
        return count, saved_files
    return count


def get_mfcc(filepath, config=None):
    '''
    将文件中的原采样数据转换为 mfcc。
    Args: 
        filepath: 保存了提前提取的采样数据的.pkl文件路径
    return: mfcc ~ (config.NUM_MFCC, frames_num)
    '''
    with open(filepath, 'rb') as f:
        y = pickle.load(f)
    if config and config.DURATION==10: # 原采样长度为20s
        y = y[:len(y)//2] # duration=10s

    n_mfcc = 20
    if config:
        n_mfcc = config.NUM_MFCC
    mfcc = librosa.feature.mfcc(y, n_mfcc=n_mfcc) # sr=22050
    return mfcc


def get_melspectrogram(filepath, config=None):
    '''
    将文件中的原采样数据转换为 melspectrogram
    Args: 
        filepath: 保存了提前提取的采样数据的.pkl文件路径
    return: mel_S ~ (config.NUM_MELS, frames_num)
    '''
    with open(filepath, "rb") as f:
        y = pickle.load(f) # duration=20s
    if config and config.DURATION==10: # 原采样长度为20s
        y = y[:len(y)//2] # duration=10s

    S = librosa.stft(y, n_fft=config.FFT_SIZE, hop_length=config.HOP_SIZE, win_length=config.WIN_SIZE)
    mel_basis = librosa.filters.mel(config.SAMPLE_RATE, n_fft=config.FFT_SIZE, n_mels=config.NUM_MELS)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T # 转置是否会有区别
    # print(mel_S.shape) # (431, 128)

    return mel_S


def get_rawmusic(filepath, half_cut=True, config=None):
    '''
    提取出文件中的原采样数据。
    Args: 
        filepath: 保存了提前提取的采样数据的.pkl文件路径
        half_cut: 原采样长度为20s，如果half_cut=True则采样长度为10s
    return: y
    '''
    with open(filepath, "rb") as f:
        y = pickle.load(f)
    if half_cut:
        y = y[:len(y)//2] 
    return y


def get_vggish(filepath, vggish, half_cut=True):
    '''
    得到音频数据的vggish嵌入表示。
    '''
    with open(filepath, "rb") as f:
        y = pickle.load(f)
    if half_cut:
        y = y[:len(y)//2] 
    embed = vggish(y, fs=22050)

    return embed



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


def get_reviews_topk_words(track_id, is_breakout, key):
    conn = MyConn()
    if key=="w_fake":
        col = "feature_words"
    elif key=="wo_fake":
        col = "feature_words_wo_fake"
    elif key=="tfidf":
        col = "feature_words_tfidf"
    elif key=="candidates":
        col = "feature_words_candidates"

    if is_breakout==1:
        bids = [r[0] for r in conn.query(sql="SELECT id FROM breakouts WHERE is_valid=1 and simi_score>=0.5 and track_id={}".format(track_id))]
        for bid in bids:
            feature_words = conn.query(sql="SELECT {} FROM breakouts_feature_words WHERE id='{}'".format(col, bid))
            if feature_words and feature_words[0][0]:
                break
    else:
        feature_words = conn.query(sql="SELECT {} FROM no_breakouts_feature_words WHERE track_id={}".format(col, track_id))
        
    if len(feature_words)>0:
        feature_words = feature_words[0][0].split()
    return feature_words


def get_reviews_vec(track_id, breakout, w2v_model, key="wo_fake"):
    '''
    指定歌曲获取评论文本向量组
    '''
    conn = MyConn()
    if key=="w_fake":
        col = "feature_words"
    elif key=="wo_fake":
        col = "feature_words_wo_fake"
    elif key=="tfidf":
        col = "feature_words_tfidf"
    elif key=="candidates":
        col = "feature_words_candidates"

    if breakout==1:
        bids = [r[0] for r in conn.query(sql="SELECT id FROM breakouts WHERE is_valid=1 and simi_score>=0.5 and track_id={}".format(track_id))]
        for bid in bids:
            feature_words = conn.query(sql="SELECT {} FROM breakouts_feature_words WHERE id='{}'".format(col, bid))
            if feature_words and feature_words[0][0]:
                break
    else:
        feature_words = conn.query(sql="SELECT {} FROM no_breakouts_feature_words WHERE track_id={}".format(col, track_id))
        
    if len(feature_words)>0:
        feature_words = feature_words[0][0].split()
    # print(breakout, feature_words)
    reviews_vec = []
    for w in feature_words:
        vec = get_w2v_vector(w, w2v_model)
        if vec is not None:
            reviews_vec.append(vec)
    return reviews_vec


def get_reviews_vec_with_freq(track_id, breakout, w2v_model, d_breakouts, d_no_breakouts, d_pos_track_breakout, with_freq=True):
    conn = MyConn()
    if breakout:
        bid = d_pos_track_breakout[track_id]
        feature_words = d_breakouts[bid]["words"]
        freqs = d_breakouts[bid]["freqs"]
        
    else:
        feature_words = d_no_breakouts[track_id]["words"]
        freqs = d_no_breakouts[track_id]["freqs"]
        
    if len(feature_words)<5:
        print(track_id, breakout)

    reviews_vec = []
    for i, w in enumerate(feature_words):
        vec= get_w2v_vector(w, w2v_model)
        if with_freq:
            vec = np.concatenate((vec, np.array([freqs[i]*100])))
        reviews_vec.append(vec)
    return reviews_vec




def resort_words_by_tfidf(words, dictionary, tfidf_model, dict_itos=None):
    '''
    将词语按照 tfidf 模型重新排序。
    '''
    bow = dictionary.doc2bow(words)  
    tfidf_sorted = sorted(tfidf_model[bow], key=lambda x:x[1], reverse=True) 
    if not dict_itos:
        stoi = dictionary.token2id
        dict_itos = dict(zip(stoi.values(), stoi.keys()))
    words_sorted = [dict_itos[r[0]] for r in tfidf_sorted]

    return words_sorted


def get_tags_vector(tags):
    '''
    获取网易云内置tags的bags向量。
    '''
    tags_pool = open("../data/metadata/自带tags.txt").read().splitlines()
    tags_d = {}
    for i, t in enumerate(tags):
        tags_d[t] = i

    vec = np.zeros(len(tags_pool))
    for t in tags:
        vec[tags_d[t]] = 1

    return vec


def get_tracks_set_db(sql, conditions):
    '''
    从数据库中获取符合条件的歌曲集
    params:
        + sql: 如 'SELECT track_id FROM tracks WHERE have_lyrics=%s'
        + conditions: 如 {"have_lyrics":1}
    return: tracks_set
    '''
    conn = MyConn()
    res = conn.query(sql=sql, conditions=conditions)
    res = set([str(r[0]) for r in res])

    return res


def get_dir_item_set(dir_name, file_postfix=".pkl"):
    '''
    获取指定文件夹下的所有项目集合
    '''
    item_set = set()
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if "DS" in file: continue
            item_set.add(file[:-len(file_postfix)])
    return item_set



def get_chorus(input_file, n_fft=2**14, clip_length=15):
    '''
    寻找歌曲副歌部分的起始时间
    param: clip_length: 指定副歌片段检测时长
    return: best_chorus_start: 副歌起始时间
    '''
    y, sr = librosa.load(input_file)
    song_length_sec = y.shape[0] / float(sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    # chorus = (chorus_start, chorus_end)
    chorus = find_chorus(chroma, sr, song_length_sec, clip_length=clip_length)

    return chorus


def get_mel_3seconds_groups(audio_file, config, offset, duration):
    '''For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

    INPUT
    
    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'

    - n_frames: length (in frames) of the input spectrogram patches.
    Data format: integer.
    Example: 187
        
    - overlap: ammount of overlap (in frames) of the input spectrogram patches.
    Note: Set it considering n_frames.
    Data format: integer.
    Example: 10
    
    OUTPUT
    
    - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
    Data format: 3D np.array (batch, time, frequency)
    
    - audio_rep: raw audio representation (spectrogram).
    Data format: 2D np.array (time, frequency)
    '''

    # 3s对应的帧数
    n_frames = librosa.time_to_frames(3, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    overlap = n_frames # 不重叠

    audio, sr = librosa.load(audio_file, sr=config.SR, offset=offset, duration=duration)
    audio_rep = librosa.feature.melspectrogram(y=audio, 
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.NUM_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch



if __name__ == '__main__':
    from MyModel.config import Config 
    get_melspectrogram("/Volumes/nmusic/NetEase2020/data/chorus_mark_rawmusic/0/101079.pkl", Config())


