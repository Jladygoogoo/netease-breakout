import os
import json
import pandas as pd
import numpy as np
import pickle
import random
import traceback
from collections import Counter
from multiprocessing import Lock as PLock
import matplotlib.pyplot as plt

from gensim import models, corpora
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from connect_db import MyConn
from threads import ThreadsGroup, ProcessGroup
from utils import assign_dir, resort_words_by_tfidf
from preprocess import tags_extractor
from fp_growth import FPGrowth


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



def add_breakouts_feature_words_to_db_LGY():
    '''
    直接在数据库中更新feature_words，使用多进程
    '''
    def task(pid, task_args):
        conn = MyConn()
        w2v_model = Word2Vec.load("../models/w2v/c4.mod")
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



def get_feature_words(text, topk=10, mode="raw", w2v_model=None, return_freq=False, candidates=None,
                    stops=None, tfidf_model=None, dictionary=None, dict_itos=None):
    '''
    提取关键词。
    params:
        text: 文本内容
        topk: 关键词的个数
        mode: 提取方式。"raw"表示直接使用词频，"stop"结合停词列表过滤再使用词频，
        "tfidf"结合tfidf_model, dictionary, dict_itos得到关键词
    '''
    if mode=="raw":
        feature_words = tags_extractor(text, topk=topk, w2v_model=w2v_model, return_freq=return_freq)
    elif mode=="stops":
        if stops==None:
            raise RuntimeError("stops can't be NONE.")
        feature_words = tags_extractor(text, topk=topk, w2v_model=w2v_model, stops_sup=stops, return_freq=return_freq)
    elif mode=="tfidf":
        feature_words = tags_extractor(text, topk=topk*2, w2v_model=w2v_model, return_freq=return_freq)
        if return_freq:
            words, freqs = zip(*feature_words)
            sorted_words = resort_words_by_tfidf(words, tfidf_model=tfidf_model, 
                                                dictionary=dictionary, dict_itos=dict_itos)
            sorted_freqs = [freqs[words.index(w)] for w in sorted_words]
            feature_words = list(zip(sorted_words, sorted_freqs))[:topk]
        else:
            feature_words = resort_words_by_tfidf(feature_words, tfidf_model=tfidf_model, 
                                                dictionary=dictionary, dict_itos=dict_itos)[:topk]

    elif mode=="candidates":
        feature_words = tags_extractor(text, topk=None, w2v_model=w2v_model, return_freq=return_freq)
        can_feature_words = []
        if return_freq:
            for p in feature_words:
                if p[0] in candidates:
                    can_feature_words.append(p)
        else:
            for w in feature_words:
                if w in candidates:
                    can_feature_words.append(w)
        feature_words = can_feature_words[:topk]

    return feature_words



def add_breakouts_feature_words_to_db():
    '''
    向breakouts_feature_words中添加数据。
    '''
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    candidates = open("../resources/music_words_cbm.txt").read().splitlines()
    # tfidf_model = models.TfidfModel.load("../models/bow/corpora_tfidf.model")
    # dictionary = corpora.Dictionary.load("../models/bow/corpora_dict.dict")
    # stoi = dictionary.token2id
    # itos = dict(zip(stoi.values(), stoi.keys()))
    rubbish_words_fake = open("../resources/rubbish_words_fake.txt").read().splitlines()
    data = conn.query(sql="SELECT id, track_id, reviews_text_path FROM breakouts WHERE is_valid=1 AND \
                                track_id IN (SELECT track_id FROM sub_tracks WHERE is_valid=1)")

    print(len(data))
    for id_, track_id, text_path in data:
        try:
            text = open(text_path).read()
            feature_words_mode = "candidates" # raw, stop, tfidf
            col = "feature_words_candidates" # feature_words, feature_words_wo_fake, feature_words_tfidf
            feature_words = get_feature_words(text, topk=10, mode=feature_words_mode, w2v_model=w2v_model, candidates=candidates, return_freq=True)
            for p in feature_words:
                print("{}:{:.3f}".format(p[0],p[1]*100), end=" ")
            print()
            if len(feature_words)<5: 
                print(track_id, "not enough words.")
                continue
            # feature_words = " ".join(feature_words)
            # conn.insert(table="breakouts_feature_words", settings={"id":id_, "track_id":track_id, col:feature_words})
            # conn.update(table="breakouts_feature_words", settings={col:feature_words}, conditions={"id":id_})
        except KeyboardInterrupt:
            break
        except:
            print(id_)
            print(traceback.format_exc())   



def add_no_breakouts_feature_words_to_db():
    '''
    向表格 no_breakouts_feature_words 中添加数据。
    '''
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    rubbish_words_fake = open("../resources/rubbish_words_fake.txt").read().splitlines()
    candidates = open("../resources/music_words_cbm.txt").read().splitlines()
    # tfidf_model = models.TfidfModel.load("../models/bow/corpora_tfidf.model")
    # dictionary = corpora.Dictionary.load("../models/bow/corpora_dict.dict")
    # stoi = dictionary.token2id
    # itos = dict(zip(stoi.values(), stoi.keys()))
    data = conn.query(sql="SELECT id, track_id, text_path FROM no_breakouts")
    d_data = {}
    for id_, track_id, text_path in data:
        if track_id in d_data:
            d_data[track_id].append((id_, text_path))
        else:
            d_data[track_id] = [(id_, text_path)]
    print(len(d_data))

    for track_id, v in d_data.items():
        try:
            text = ""
            for id_, text_path in v:
                text += open(text_path).read()
            feature_words_mode = "candidates" # raw, stop, tfidf
            col = "feature_words_candidates" # feature_words, feature_words_wo_fake, feature_words_tfidf
            feature_words = get_feature_words(text, topk=10, mode=feature_words_mode, w2v_model=w2v_model, candidates=candidates, return_freq=True)          
            for p in feature_words:
                print("{}:{:.3f}".format(p[0],p[1]*100), end=" ")
            print()            
            if len(feature_words)<5: 
                print(track_id, "not enough words.")
                continue
            # feature_words = " ".join(feature_words)
            # conn.insert(table="no_breakouts_feature_words", settings={"id":id_, "track_id":track_id, col:feature_words})
            # conn.update(table="no_breakouts_feature_words", settings={col:feature_words}, conditions={"track_id":track_id})
        except KeyboardInterrupt:
            break
        except:
            print(track_id)
            print(traceback.format_exc())




def add_breakouts_feature_words_to_json():
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    candidates = open("../resources/music_words/music_words_cls_pos_pred.txt").read().splitlines()
    # remove = open("../resources/music_words/music_words_similar.txt").read().splitlines()
    # candidates = [w for w in candidates if w not in remove]
    data = conn.query(sql="SELECT id, track_id, reviews_text_path FROM breakouts WHERE is_valid=1 AND \
                                track_id IN (SELECT track_id FROM sub_tracks WHERE is_valid=1)")


    json_data = {}
    for id_, track_id, text_path in data:
        try:
            text = open(text_path).read()
            feature_words_mode = "candidates" # raw, stop, tfidf
            feature_words = get_feature_words(text, topk=10, mode=feature_words_mode, w2v_model=w2v_model, candidates=candidates, return_freq=True)          
            words, freqs = zip(*feature_words)
            json_data[id_] = {"words":words, "freqs":freqs, "len": len(words)}
            if len(feature_words)<5: 
                print(id_, "not enough words.")
        except KeyboardInterrupt:
            break
        except:
            print(track_id)
            print(traceback.format_exc())
    with open("../data/reviews_feature_words_with_freqs/breakouts_cls.json", 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)




def add_no_breakouts_feature_words_to_json():
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    rubbish_words_fake = open("../resources/rubbish_words_fake.txt").read().splitlines()
    candidates = open("../resources/music_words/music_words_cls_pos_pred.txt").read().splitlines()
    # remove = open("../resources/music_words/music_words_similar.txt").read().splitlines()
    # candidates = [w for w in candidates if w not in remove]
    data = conn.query(sql="SELECT id, track_id, text_path FROM no_breakouts")
    d_data = {}
    for id_, track_id, text_path in data:
        if track_id in d_data:
            d_data[track_id].append((id_, text_path))
        else:
            d_data[track_id] = [(id_, text_path)]
    print(len(d_data))

    json_data = {}
    for track_id, v in list(d_data.items()):
        try:
            text = ""
            for id_, text_path in v:
                text += open(text_path).read()
            feature_words_mode = "candidates" # raw, stop, tfidf
            feature_words = get_feature_words(text, topk=10, mode=feature_words_mode, w2v_model=w2v_model, candidates=candidates, return_freq=True)          
            words, freqs = zip(*feature_words)
            json_data[track_id] = {"words":words, "freqs":freqs, "len":len(words)}
            if len(feature_words)<5: 
                print(track_id, "not enough words.")
        except KeyboardInterrupt:
            break
        except:
            print(track_id)
            print(traceback.format_exc())
    with open("../data/reviews_feature_words_with_freqs/no_breakouts_cls.json", 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)



def tnse():
    music_words_pos = open("../resources/music_words/music_words_cls_pos_pred.txt").read().splitlines()
    music_words_neg = open("../resources/music_words/music_words_cls_neg_pred.txt").read().splitlines()
    music_words = music_words_pos + music_words_neg
    # music_words = open("../resources/music_words/music_words_cbm.txt").read().splitlines()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    music_words_vec = [w2v_model.wv[w] for w in music_words]
    tnse_model = TSNE(perplexity=3.0)
    Y = tnse_model.fit_transform(music_words_vec)
    for i in range(len(Y)):
        if Y[i][0]<=-100 or Y[i][0]>=90 or Y[i][1]<=-100 or Y[i][1]>=90:
            print(music_words[i])

    plt.scatter(Y[:,0], Y[:,1])
    plt.title("music_words_netease_seed")
    plt.show()


def music_words_classifier_select():
    pos_words = open("../resources/music_words/music_words_cls_pos.txt").read().splitlines()
    neg_words = open("../resources/music_words/music_words_cls_neg.txt").read().splitlines()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    X = [w2v_model.wv[w] for w in pos_words] + [w2v_model.wv[w] for w in neg_words]
    y = [1]*len(pos_words) + [0]*len(neg_words)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, shuffle=True)
    print("train dataset: pos({})/total({})={:.2f}".format(sum(y_train), len(y_train), sum(y_train)/len(y_train)))
    print("test dataset: pos({})/total({})={:.2f}".format(sum(y_test), len(y_test), sum(y_train)/len(y_train)))

    models = [
      SVC(gamma=1, C=0.1),
      RandomForestClassifier(max_depth=5, n_estimators=10),
      LogisticRegression(solver="lbfgs")
    ]
    print("train:")
    for model in models:
        m_name = model.__class__.__name__
        accs = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        acc = accs.mean()
        print("{}: {:.3f}".format(m_name, acc))

    # 得到逻辑回归的效果最好 acc=0.907
    model = LogisticRegression(solver="lbfgs").fit(X_train, y_train)
    pred_test = model.predict(X_test)
    corrects = sum([1 if pred_test[i]==y_test[i] else 0 for i in range(len(pred_test))])
    print("test:")
    print("{}: {:.3f}".format(model.__class__.__name__, corrects/len(y_test)))


def music_words_from_classifier():
    '''
    使用分类器得到music_words
    '''
    # 使用全部的人工数据训练逻辑回归模型
    pos_words = open("../resources/music_words/music_words_cls_pos.txt").read().splitlines()
    neg_words = open("../resources/music_words/music_words_cls_neg.txt").read().splitlines()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    X = [w2v_model.wv[w] for w in pos_words] + [w2v_model.wv[w] for w in neg_words]
    y = [1]*len(pos_words) + [0]*len(neg_words)
    model = LogisticRegression(solver="lbfgs").fit(X, y)

    # 对关键词进行分类
    with open("../resources/feature_words_counter_d.pkl", "rb") as f:
        d_words_freq = pickle.load(f)
    # print(len(d_words_freq))
    # size = 13392
    # print(np.median(list(d_words_freq.values())))
    median = 9.470145366731379e-06
    pos_words_pred = []
    neg_words_pred = []
    for word, freq in d_words_freq.items():
        if freq<=median or not w2v_model.wv.__contains__(word): 
            continue
        vec = w2v_model.wv[word]
        word_label_prob = model.predict_proba([vec])[0][1]
        if word_label_prob<=0.3:
            neg_words_pred.append(word)
        elif word_label_prob>=0.8:
            pos_words_pred.append(word)
        else:
            continue

    print(len(pos_words_pred), len(neg_words_pred))
    with open("../resources/music_words/music_words_cls_pos_pred.txt", 'w') as f:
        f.write("\n".join(pos_words_pred))
    with open("../resources/music_words/music_words_cls_neg_pred.txt", 'w') as f:
        f.write("\n".join(neg_words_pred))




if __name__ == '__main__':
    # add_feature_words_to_db()
    # rubbish_tags()
    # feature_words_counter()
    # special_words()
    # co_occurance()
    # add_feature_words_to_db()
    # filtered_feature_words()
    # add_breakouts_feature_words_to_json()
    # add_no_breakouts_feature_words_to_json()
    # add_breakouts_feature_words_to_db()
    tnse()
    # music_words_classifier()
    # music_words_from_classifier()
