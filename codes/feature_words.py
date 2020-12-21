import os
import json
import pandas as pd
import numpy as np
import pickle
import random
import traceback
from collections import Counter
from multiprocessing import Lock as PLock

from gensim.models import Word2Vec

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from utils import assign_dir
from preprocess import tags_extractor
from fp_growth import FPGrowth


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




def breakouts_cmp_nonbreakouts():
    '''
    考察 breakouts 和 no_breakouts 关键词的重叠
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



def feature_words_counter():
    '''
    + 关键词词频的统计
    + 由初步此筛选出special_words和rubbish_words
    + 将feature_words_counter字典保存为pkl，以便后续使用
    '''

    conn = MyConn()
    words_counter = Counter()

    for res in conn.query(targets=["feature_words"], table="breakouts_feature_words_c3"):
        feature_words = res[0].split()
        words_counter.update(feature_words)
    words_counter_size = sum(words_counter.values())
    words_counter_d = dict(map(lambda p:(p[0], p[1]/words_counter_size), words_counter.most_common()))
    # for k,v in words_counter.most_common()[:500]:
    #     print(k, v)

    with open("../resources/feature_words_counter_d.pkl", "wb") as f:
        pickle.dump(words_counter_d, f)
    


def rubbish_tags():
    '''
    + 统计每个爆发点关键词中rubbish_tags的个数
    + 将垃圾标签占比大的样本点作为噪声筛去
    + 将关键词中的垃圾标签删除并上传至数据库
    '''

    rubbish = open("../resources/rubbish_tags.txt").read().splitlines()
    conn = MyConn()

    records = []
    for res in conn.query(targets=["id", "feature_words"], table="breakouts_feature_words_c3"):
        # if conn.query(table="breakouts", targets=["release_drive"], fetchall=False, conditions={"id": res[0]})[0] == 1:
        #     continue
        feature_words = res[1].split()
        rubbish_count = 0
        for w in feature_words:
            if w in rubbish: 
                rubbish_count += 1
        records.append([res[0], rubbish_count, feature_words])

    records.sort(key=lambda x:x[1], reverse=True)
    for r in records:
        print(r)

    # for r in records:
    #     if r[1] >= 7:
    #         conn.update(table="breakouts", settings={"fake": 1}, conditions={"id": r[0]})




def filtered_feature_words():
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c3.mod")
    rubbish = open("../resources/rubbish_words.txt").read().splitlines()
    with open("../resources/feature_words_counter_d.pkl", "rb") as f:
        feature_words_counter_d = pickle.load(f)

    for id_, text in conn.query(targets=["id", "feature_words"], table="breakouts_feature_words_c3"):
        # 过滤掉rubbish_words
        feature_words = text.split()
        filtered_feature_words = []
        for w in feature_words:
            if w not in rubbish:
                filtered_feature_words.append(w)

        # conn.update(table="breakouts_feature_words_c3", 
        #     settings={"filtered_feature_words": " ".join(filtered_feature_words)},
        #     conditions={"id": id_})

        # 基于w2v模型合并意思相近的词
        simi_matrix = []
        size = len(filtered_feature_words)
        for i in range(size):
            tmp = []
            for j in range(size):
                w1, w2 = filtered_feature_words[i], filtered_feature_words[j]
                tmp.append(w2v_model.wv.similarity(w1, w2))
            simi_matrix.append(tmp)

        selected = [0]*size
        groups = []
        for i in range(size):
            if selected[i]==1: 
                continue
            selected[i] = 1
            group_wi = [i]
            for j in range(size):
                if selected[j]!=1:
                    for wi in group_wi:
                        if simi_matrix[wi][j]>=0.55:
                            group_wi.append(j)
                            selected[j] = 1
                            break
            group = [filtered_feature_words[wi] for wi in group_wi]
            groups.append(group)

        # print(groups)
        clean_feature_words = []
        for g in groups:
            if len(g)>1:
                clean_feature_words.append(g[np.argmax([feature_words_counter_d[w] for w in g])])
            else:
                clean_feature_words.append(g[0])
        # print(clean_feature_words)

        conn.update(table="breakouts_feature_words_c3", 
            settings={"clean_feature_words": " ".join(clean_feature_words)},
            conditions={"id": id_})






def special_words():
    '''
    基于给定的special_words集合对关键词分类
    '''
    conn = MyConn()
    special_words = {
        "高考": ["高考", "考研", "上岸"], 
        "节日": ["圣诞", "圣诞节", "圣诞快乐", "新年", "新年快乐", "情人节", "七夕", "七夕节", "中秋", "中秋节", "清明", "清明节", "元宵", "元宵节", "儿童节", "母亲节", "愚人节", "万圣节", "父亲节", "光棍节", "国庆", "国庆节"],
        "微博": ["热搜", "微博", "微博来"],
        "抖音": ["抖音", "快手"],
        "b站": ["b站"],
        "bgm": ["剪辑", "bgm", "预告", "鬼畜"],
        "收费": ["收费", "vip"],
        "翻唱": ["原版", "原唱", "翻唱", "版本", "原曲", "完整版"],
        "rip": ["rip", "天堂"],
        "网易云操作": ["日推", "日签", "乐签", "签到"],
        "生日快乐": ["生日快乐"],
        "抄袭": ["抄袭"],
        "版权": ["版权"]
    }

    def update_db():
        data = conn.query(targets=["id", "filtered_feature_words"], table="breakouts_feature_words_c3")
        for id_, text in data:
            filtered_feature_words = text.split()
            hit_special_words = set()
            for w in filtered_feature_words:
                for k, v in special_words.items():
                    if w in v:
                        hit_special_words.add(k)
                        break
            hit_special_words = " ".join(list(hit_special_words))
            conn.update(table="breakouts_feature_words_c3", settings={"special_words": hit_special_words}, conditions={"id": id_})


    def to_be_filtered():
        filter_words_d = {"收费":0, "网易云操作":0, "版权":0}
        filter_breakouts = set()
        for id_, special_words in conn.query(targets=["id", "special_words"], table="breakouts_feature_words_c3"):
            for w in special_words.split():
                if w in filter_words_d:
                    filter_words_d[w] += 1
                    filter_breakouts.add(id_)
        for k, v in filter_words_d.items():
            print(k, v)
        print(len(filter_breakouts))
        for id_ in filter_breakouts:
            conn.update(table="breakouts", settings={"capital_drive": 1}, conditions={"id": id_})


    # update_db()
    to_be_filtered()



def co_occurance():
    conn = MyConn()

    def within_one_breakout(special_words_only=True):
        # print(len(data))

        if not special_words_only: # 不限于special words
            data = [r[0].split() for r in conn.query(targets=["filtered_feature_words"], table="breakouts_feature_words_c2")]
            data = list(filter(lambda x:len(x)>=4, data))
            fp_growth_model = FPGrowth(min_support=5)
            fp_growth_model.train(data)

        else: # 只考察special words
            data = [r[0].split() for r in conn.query(targets=["special_words"], table="breakouts_feature_words_c2")]
            data = list(filter(lambda x:len(x)>=2, data))
            fp_growth_model = FPGrowth(min_support=1)
            fp_growth_model.train(data)


    def within_one_track():
        data = conn.query(targets=["id", "special_words"], table="breakouts_feature_words_c2")
        data = list(filter(lambda p:len(p[1])>=1, data))
        data = [(p[0], p[1].split()[0]) for p in data]
        print(len(data))

        track_2_special_words = {}
        for id_, special_word in data:
            track_id = id_.split('-')[0]
            if track_id in track_2_special_words:
                track_2_special_words[track_id].append(special_word)
            else:
                track_2_special_words[track_id] = [special_word]

        for k, v in track_2_special_words.items():
            conn.update(table="breakout_tracks_complements", settings={"special_words": ' '.join(v)}, conditions={"track_id":k})

        fp_growth_model = FPGrowth(min_support=1)
        fp_growth_model.train(list(track_2_special_words.values()))

    # within_one_breakout()
    within_one_track()






if __name__ == '__main__':
    # add_feature_words_to_db()
    # rubbish_tags()
    # feature_words_counter()
    special_words()
    # co_occurance()
    # add_feature_words_to_db()
    # filtered_feature_words()
