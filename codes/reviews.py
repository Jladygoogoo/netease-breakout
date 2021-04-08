import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from gensim import corpora, models
from gensim.models import Word2Vec
from queue import Queue
from threading import Thread, Lock

from preprocess import tags_extractor
from connect_db import MyConn
from threads import ThreadsGroup
from breakout_tools import get_breakout_index_seq, get_reviews_df, get_reviews_count, get_no_breakouts, get_breakouts_text
from utils import get_every_day, assign_dir, count_files
from w2v import words_simi_score
from feature_words import get_feature_words

def all_reviews_before_breakout():
    '''
    将所有爆发点的预评论文本保存至本地。
    '''
    conn = MyConn()
    sql = "SELECT track_id, json_path FROM tracks WHERE lyrics_path is not null and mp3_path is not null and json_path is not null"
    d_tid_reviews_jsonpath = dict(conn.query(sql=sql))
    data = conn.query(table="breakouts", targets=["id", "track_id", "date"], conditions={"is_valid":1})

    def _reviews_before_breakout(track_id, filepath, bdate, days_thres=7, count_thres=500):
        '''
        提取单个爆发点的预评论
        '''
        df = get_reviews_df(filepath)
        reviews_count, dates = get_reviews_count(df["date"].values)
        scaled_bis = get_breakout_index_seq(reviews_count)

        d_date_bi = dict(zip(dates, scaled_bis))
        bdate = datetime.strftime(bdate,'%Y-%m-%d')
        right = dates.index(bdate)
        left = right
        while right>days_thres-1: 
            right -= 1
            # 此处的25根据判断爆发的beta=50来设定
            if scaled_bis[right]<=25:
                # 至少收集一周的信息，要求1周内不出现爆发点
                if np.sum(np.array(scaled_bis[right-days_thres+1:right])<=25)==days_thres-1:
                    break
        if right<days_thres-1: return # 不符合提取要求

        left = right-days_thres+1
        sum_reviews_num = np.sum(reviews_count[left:right+1])
        while sum_reviews_num<500 and left>=0:
            left -= 1
            if scaled_bis[left]<=25:
                sum_reviews_num += reviews_count[left]
        if sum_reviews_num<500: return # 不符合提取要求

        left_date, right_date = dates[left], dates[right]
        # print(left_date, right_date)

        reviews_text = []
        valid_dates = get_every_day(left_date, right_date)        
        for i in range(len(df)):
            if df.iloc[i]["date"] in valid_dates:
                reviews_text.append(df.iloc[i]["content"])

        return "\n".join(reviews_text)

        # new_df = pd.DataFrame(zip(reviews_count, scaled_bis), columns=["reviews_num", "breakout_index"])
        # new_df.to_csv("../results/reviews_num_csv/{}_bi.csv".format(track_id))

    base_dir = "/Volumes/nmusic/NetEase2020/data/reviews_before_breakout"
    print(len(data))
    for i, item_data in enumerate(data):
        if i<1100: continue # 上一次
        breakout_id, track_id, bdate = item_data
        filepath = d_tid_reviews_jsonpath[track_id]
        write_dir = os.path.join(base_dir, str(i//100))
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        try:
            reviews_text = _reviews_before_breakout(track_id, filepath, bdate)
        except:
            print(i, breakout_id, "ERROR")
            continue
        if not reviews_text: 
            print(i, breakout_id)
            continue
        with open(os.path.join(write_dir, "{}.txt".format(breakout_id)), 'w') as f:
            f.write(reviews_text)



def compare_reviews_before_and_within_breakout():
    '''
    比较预评论和爆发评论。
    '''
    stops_sup = open("../resources/rubbish_words_tags.txt").read().splitlines()
    conn = MyConn()
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    data = conn.query(sql="SELECT id, reviews_text_path, before_reviews_text_path FROM breakouts WHERE before_reviews_text_path is not null AND is_valid=1 AND simi_score is null")

    for id_, p1, p2 in data:
        text1 = "\n".join(open(p1).read().splitlines()[:1000])
        text2 = "\n".join(open(p2).read().splitlines()[:1000])
        feature_words1 = tags_extractor(text1, topk=5, stops_sup=stops_sup, w2v_model=w2v_model)
        feature_words2 = tags_extractor(text2, topk=5, stops_sup=stops_sup, w2v_model=w2v_model)
        simi_score = words_simi_score(feature_words1, feature_words2, w2v_model)
        conn.update(table="breakouts", settings={"simi_score":float(simi_score)}, conditions={"id":id_})
        print(id_, "simi_score: {:.2f}".format(simi_score))
        print(feature_words1)
        print(feature_words2)


def prep_neg_reviews_text():
    '''
    准备非爆发歌曲（负样本）的评论文本数据
    '''
    conn = MyConn()
    save_dir_prefix = "/Volumes/nmusic/NetEase2020/data/no_breakouts_text"
    n_dir, dir_size = 2, 100
    neg_tracks = [r[0] for r in conn.query(sql="SELECT track_id FROM sub_tracks WHERE valid_bnum=0 AND is_valid=1")]
    print(len(neg_tracks))
    flag, saved_files = count_files(save_dir_prefix, return_files=True) # 已经提取好的歌曲
    saved_tracks = [x[:-6] for x in saved_files]

    q_tracks = Queue()
    for t in neg_tracks: 
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
                task_args["flag"] += 1
                lock.release()

                reviews_json_path = conn.query(table="sub_tracks", targets=["json_path"], conditions={"track_id":tid}, fetchall=False)[0]

                df = get_reviews_df(reviews_json_path)
                reviews_count, dates = get_reviews_count(df["date"].values)
                no_breakouts_group = get_no_breakouts(reviews_count, min_reviews=200, thres=25)

                # 抽样
                samples = np.floor(np.linspace(0, len(no_breakouts_group)-1, min(len(no_breakouts_group), 3)))
                no_breakouts_group = [no_breakouts_group[int(s)] for s in samples]

                if len(no_breakouts_group)>0:
                    for flag, group in enumerate(no_breakouts_group):
                        # 基本信息上传至数据库
                        # group: (left_index, right_index, reviews_acc)
                        data = {
                            "id": '-'.join([tid, str(flag)]),
                            "track_id": tid,
                            "flag": flag,
                            "start_date": dates[group[0]],
                            "end_date": dates[group[1]],
                            "reviews_acc_num": group[2]
                        }
                        conn.insert(table="no_breakouts", settings=data)

                        # 将文本保存到本地
                        text = ""
                        for point in range(group[0], group[1]):
                            date = dates[point]
                            text += get_breakouts_text(df, date)
                        with open(os.path.join(dirpath, "{}-{}.txt".format(tid, flag)), 'w') as f:
                            f.write(text)
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



def build_tfidf():
    conn = MyConn()
    w2v_model = models.Word2Vec.load("../models/w2v/c4.mod")
    corpus_source = ["/Volumes/nmusic/NetEase2020/data/breakouts_text", "/Volumes/nmusic/NetEase2020/data/no_breakouts_text_LGY"]
    filepath_set = []
    for src in corpus_source:
        for root, dirs, files in os.walk(src):
            for file in files:
                if "OS" in file: continue
                filepath_set.append(os.path.join(root, file))

    print(len(filepath_set))
    docs = []
    for i, filepath in enumerate(filepath_set):
        text = open(filepath).read()
        try:
            docs.append(tags_extractor(text, topk=20, w2v_model=w2v_model))
        except KeyboardInterrupt:
            break
        except:
            print(traceback.format_exc())
        if i%100==0:
            print("{} docs processed.".format(i))

    dictionary = corpora.Dictionary(docs)
    bows = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bows)

    dictionary.save('../models/bow/corpora_dict.dict') # 重载用corpora.Dictionary.load(path)
    tfidf.save('../models/bow/corpora_tfidf.model') # 重载用models.TfidfModel.load(path)


def music_words_from_netease_tags():
    '''
    从网易云提供的72种标签，得到距离相近的词
    '''
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    tags72 = open("../data_related/自带tags.txt").read().splitlines()
    music_words_set = set()
    layer1 = set()
    layer2 = set()
    for t in tags72:
        t = t.lower().replace('_','-')
        ts = t.split('/')
        for t in ts:
            if w2v_model.wv.__contains__(t):
                music_words_set.add(t)
                for w, s in w2v_model.wv.most_similar([t], topn=15):
                    layer1.add(w)
    for w in layer1:
        for w,s in w2v_model.wv.most_similar([w], topn=2):
            layer2.add(w)

    music_words_set = music_words_set.union(layer1.union(layer2))
    print(len(music_words_set))
    print(music_words_set)
    with open("../resources/music_words/music_words_from_netease_tags_2.txt", "w") as f:
        f.write("\n".join(music_words_set))


def music_words_all():
    '''
    网易云种子（衍生） + 自定义词（衍生）
    '''
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    # music_words_netease
    music_words_set = set(open("../resources/music_words/music_words_from_netease_tags_2.txt").read().splitlines())  
    # music_words_self_sup
    music_words_self_sup = open("../resources/music_words/music_words_from_self_sup.txt").read().splitlines()
    for sw in music_words_self_sup:
        music_words_set.add(sw)
        for w,s in w2v_model.wv.most_similar([sw], topn=3):
            music_words_set.add(w)
    with open("../resources/music_words/music_words_cbm2.txt", "w") as f:
        f.write("\n".join(music_words_set))



def test_music_words():
    '''
    【测试】筛选出来的music_words的可行性
    '''
    conn = MyConn()
    data = conn.query(sql="SELECT id, reviews_text_path FROM breakouts WHERE is_valid=1 AND \
                                track_id IN (SELECT track_id FROM sub_tracks WHERE is_valid=1)")
    w2v_model = Word2Vec.load("../models/w2v/c4.mod")
    stops = open("../resources/rubbish_words_fake.txt").read().splitlines()
    candidates = open("../resources/music_words_from_netease_tags.txt").read().splitlines()


    for id_, path in data:
        text = open(path).read()
        feature_words = get_feature_words(text, topk=10, mode="stops", w2v_model=w2v_model, stops=stops)
        music_words = get_feature_words(text, topk=10, mode="candidates", w2v_model=w2v_model, candidates=candidates)
        # print(feature_words)
        # print(music_words)
        # print()
        if len(music_words)<5:
            print(id_, music_words)



if __name__ == '__main__':
    # all_reviews_before_breakout()
    # compare_reviews_before_and_within_breakout()
    # prep_neg_reviews_text()
    # build_tfidf()
    # music_words_from_netease_tags()
    # test_music_words()
    music_words_all()
    # music_words_from_netease_tags()








