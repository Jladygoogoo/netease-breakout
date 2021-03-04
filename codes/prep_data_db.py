import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import traceback
import random
import logging
from threading import Thread, Lock
from multiprocessing import Lock as PLock

import librosa
from gensim.models import Word2Vec
import soundfile as sf

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from utils import get_chorus


def copy_columns(t1, t2, col, key_col="track_id"):
    '''
    在数据库中，将一张表某列的信息复制到另一张表
    params:
        t1: 被拷贝的表
        t2: 被粘贴的表
        col: 列名称
        key_col: 键值
    '''
    conn = MyConn()
    data = conn.query(table=t1, targets=[key_col, col])
    for key_v, v in data:
        try:
            conn.update(table=t2, settings={col:v}, conditions={key_col:key_v})
        except:
            print("ERROR {}: {}".format(key_col, key_v))



def update_chorus_start():
    def task(thread_id, task_args):
        '''
        检测 chorus_start 并上传至数据库（tracks & sub_tracks）
        '''
        conn = MyConn()
        while 1:
            sql = "SELECT track_id, mp3_path from sub_tracks WHERE chorus_start IS NULL AND valid_bnum=0"
            task_args["lock"].acquire()
            res = conn.query(sql=sql, fetchall=False)
            if not res: 
                break # sub_tracks表格中不存在chorus_start is null的歌曲
            tid, mp3_path = res[0], res[1]
            conn.update(table="sub_tracks", settings={"chorus_start":0}, conditions={"track_id":tid})
            task_args["lock"].release()

            try:
                # 检测并截取歌曲的副歌部分，设置时长为20s
                duration = 20
                chorus = get_chorus(mp3_path, clip_length=duration)
                if chorus is None:
                    conn.update(table="sub_tracks", settings={"chorus_start":-1}, conditions={"track_id":tid})
                else:
                    chorus_start, chorus_end = chorus
                    conn.update(table="sub_tracks", settings={"chorus_start":chorus_start}, 
                            conditions={"track_id":tid})
                    conn.update(table="tracks", settings={"chorus_start":chorus_start}, 
                            conditions={"track_id":tid})
            except KeyboardInterrupt:
                conn.update(sql="UPDATE sub_tracks SET chorus_start=NULL WHERE track_id={}".format(tid))
                break
            except:
                print(tid, traceback.format_exc())

    # 开启thread群工作
    lock = Lock()
    task_args = {"lock":lock}
    threads_group = ThreadsGroup(task=task, n_thread=4, task_args=task_args)
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
            "artist": ",".join([item["name"] for item in content["songs"][0]["ar"]]),
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



def create_subtracks_table():
    '''
    创建sub_tracks表格。
    歌曲筛选条件：
        + 拥有mp3_path,lyrics_path,json_path(reviews)
        + 对于有爆发点的歌，要求爆发点不属于fake,capital_drive,release_drive，爆发点的reviews_num>=100,beta>=50
        + 拥有artist_vec
    '''
    # 读取artists的向量表示
    with open("../data/artists_vec_dict.pkl", "rb") as f:
        d_artist_vec = pickle.load(f)
    conn = MyConn()

    # 拥有mp3_path,lyrics_path,json_path(reviews)
    data = conn.query(sql="SELECT track_id, bnum, mp3_path, lyrics_path, json_path FROM tracks WHERE\
                           bnum IS NOT NULL and mp3_path IS NOT NULL and lyrics_path IS NOT NULL and json_path IS NOT NULL")

    # 要求爆发点不属于fake,capital_drive,release_drive，爆发点的reviews_num>=100,beta>=50
    # d_track_valid_bnum记录歌曲中valid_breakouts的数量
    targets = ("id", "track_id", "beta", "reviews_num", "release_drive", "capital_drive", "fake")
    breakouts = conn.query(table="breakouts", targets=targets)
    d_track_valid_bnum = {}
    for b in breakouts:
        d_tmp = dict(zip(targets, b))
        if d_tmp["beta"]>=50 and d_tmp["reviews_num"]>=100 and \
            d_tmp["release_drive"]+d_tmp["capital_drive"]+d_tmp["fake"]==0:
            tid = d_tmp["track_id"]
            if tid in d_track_valid_bnum:
                d_track_valid_bnum[tid] += 1
            else:
                d_track_valid_bnum[tid] = 1

    new_data = []
    for item in data:
        track_id, bnum = item[0], item[1]
        # 用valid_breakouts筛选
        valid_bnum = 0
        if bnum>0:
            if track_id not in d_track_valid_bnum: 
                continue 
            valid_bnum = d_track_valid_bnum[track_id]

        # 用artist_vec筛选
        valid_artist = None
        artists = conn.query(table="details", targets=["artists"], conditions={"track_id":track_id}, fetchall=False)
        if artists:
            artists = artists[0].split(',')
            for ar in artists:
                if ar.lower().strip() in d_artist_vec: 
                    valid_artist = ar.lower().strip()
                    break
        if not valid_artist: continue
        new_data.append([track_id, valid_bnum, valid_artist, item[2], item[3], item[4]])

    # 提交至数据库
    columns = ("track_id", "valid_bnum", "artist", "mp3_path", "lyrics_path", "json_path")
    for item in new_data:
        conn.insert(table="sub_tracks", settings=dict(zip(columns, item)))






def refine_subtracks():
    '''
    进一步精炼sub_tracks表格。
    + 将只存在非法爆发点的歌曲除去（release_drive / capital drive / fake）
    + 要求存在 beta>=50&&reviews_num>=100 的爆发点
    '''
    conn = MyConn()
    targets = ("id", "track_id", "beta", "reviews_num", "release_drive", "capital_drive", "fake")
    breakouts = conn.query(table="breakouts", targets=targets)
    d_track_valid_bnum = {}

    for b in breakouts:
        d_tmp = dict(zip(targets, b))
        if d_tmp["beta"]>=50 and d_tmp["reviews_num"]>=100 and \
            d_tmp["release_drive"]+d_tmp["capital_drive"]+d_tmp["fake"]==0:
            tid = d_tmp["track_id"]
            if tid in d_track_valid_bnum:
                d_track_valid_bnum[tid] += 1
            else:
                d_track_valid_bnum[tid] = 1

    subtracks = [r[0] for r in conn.query(sql="SELECT track_id FROM sub_tracks WHERE bnum>0")]
    count_valid = 0
    for tid in subtracks:
        if tid in d_track_valid_bnum:
            count_valid += 1
        else:
            # print(tid, end=", ")            
            conn.delete(table="sub_tracks", conditions={"track_id":tid})
    print("\n", count_valid)



def update_path(name, root_dir, offset, table):
    '''
     在数据库中更新路径
    '''
    conn = MyConn()
    count_update = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files: 
            if "OS" in file: continue
            filepath = os.path.join(root, file)
            track_id = file.split('/')[-1][:-offset]
            res = conn.query(table=table, targets=[name], conditions={"track_id":track_id}, fetchall=False)
            if res and res[0] is None:
                conn.update(table=table, settings={name:filepath}, conditions={"track_id":track_id})
                count_update += 1
    print(count_update)



def add_feature_words_to_db():
    '''
    直接在数据库中更新feature_words，使用多进程
    '''
    def task(pid, task_args):
        conn = MyConn()
        w2v_model = Word2Vec.load("../models/w2v/c3.mod")
        while 1:
            task_args["lock"].acquire()
            res = conn.query(targets=["id", "text_path"], conditions={"have_words":0}, 
                    table="breakouts", fetchall=False)
            if res is not None:
                id_, text_path = res
                conn.update(table="breakouts",
                            settings={"have_words": 1},
                            conditions={"id": id_})
                task_args["lock"].release()

                try:
                    feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
                    conn.insert(table="breakouts_feature_words_c3",
                                settings={"id":id_, "feature_words":" ".join(feature_words)})

                    # print("[Process-{}] id: {}, feature_words: {}".format(pid, id_, feature_words))
                except:
                    conn.update(table="breakouts",
                                settings={"have_words": 0},
                                conditions={"id": id_})
                    print(id_)
                    print(traceback.format_exc())
                    break

            else:
                task_args["lock"].release()
                break

    lock = PLock()
    task_args = {"lock":lock}
    process_group = ProcessGroup(task=task, n_procs=1, task_args=task_args)
    process_group.start()



def add_feature_words_to_db_2():
    '''
    直接在数据库中更新feature_words，使用多进程。对应no_breakouts。
    '''
    def task(pid, task_args):
        conn = MyConn()
        w2v_model = Word2Vec.load("../models/w2v/b1.mod")
        while 1:
            task_args["lock"].acquire()
            res = conn.query(sql="SELECT id,text_path from no_breakouts WHERE have_words=0 AND text_path IS NOT NULL", 
                    table="no_breakouts", fetchall=False)
            if res is not None:
                id_, text_path = res
                if text_path is None: 
                    task_args["lock"].release()
                    continue
                conn.update(table="no_breakouts",
                            settings={"have_words": 1},
                            conditions={"id": id_})
                task_args["lock"].release()

                try:
                    feature_words = tags_extractor(open(text_path).read(), topk=10, w2v_model=w2v_model)
                    # print(text_path)
                    # print(open(te xt_path).read())
                    conn.insert(table="no_breakouts_feature_words_1",
                                settings={"id":id_, "feature_words":" ".join(feature_words)})

                    print("[Process-{}] id: {}, feature_words: {}".format(pid, id_, feature_words))
                except KeyboardInterrupt:
                    conn.update(table="no_breakouts",
                                settings={"have_words": 0},
                                conditions={"id": id_})
                    return
                except TypeError:
                    print("TypeError", text_path)
                    continue
                except:
                    print(traceback.format_exc())
                    conn.update(table="no_breakouts",
                                settings={"have_words": 0},
                                conditions={"id": id_})
                    continue
                    

            else:
                task_args["lock"].release()
                break

    lock = PLock()
    task_args = {"lock":lock}
    process_group = ProcessGroup(task=task, n_procs=3, task_args=task_args)
    process_group.start()



def update_special_tag1():
    '''
    从之前生成的special_words表格中更新sub_tracks中的special_tag
    '''
    conn = MyConn()
    sql = "SELECT id, special_words FROM breakouts_feature_words_c3 WHERE LENGTH(special_words)>0"
    data = conn.query(sql=sql)
    special_words = ["高考", "翻唱", "节日", "抖音"]
    for bid, text in data:
        words = text.split()
        for w in words:
            if w in special_words:
                tid = bid.split('-')[0]
                try:
                    conn.update(table="sub_tracks", settings={"special_tag":w}, conditions={"track_id":tid})
                except:
                    print(tid)
                break



if __name__ == '__main__':
    # update_chorus_start()
    # create_subtracks_table()
    # update_path(name="rawmusic_path", root_dir="/Volumes/nmusic/NetEase2020/data/chorus_mark_rawmusic/", offset=4, table="sub_tracks")
    # refine_subtracks()
    copy_columns(t1="tracks", t2="sub_tracks", col="language")
    # update_special_tag1()
