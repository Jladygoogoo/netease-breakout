import os
import time
import json
import pandas as pd 
import numpy as np
from connect_db import MyConn
import random
random.seed(21)
from collections import Counter
from gensim.models import Doc2Vec

from preprocess import cut, replace_noise
from utils import cosine_similarity


def check_feature_words():
    conn = MyConn()
    breakouts_feature_words = Counter()
    res = [r[0].split() for r in conn.query(targets=["feature_words"], table="breakouts_feature_words_1")]
    for r in res:
      breakouts_feature_words.update(r)

    valid_breakouts_feature_words = [p[0] for p in filter(lambda x:x[1]>=30, breakouts_feature_words.most_common())]

    # no_breakouts_feature_words = Counter()
    # res = [r[0].split() for r in conn.query(targets=["feature_words"], table="no_breakouts_feature_words_1")]
    # for r in res:
    #   no_breakouts_feature_words.update(r)

    # valid_no_breakouts_feature_words = [p[0] for p in filter(lambda x:x[1]>=30, no_breakouts_feature_words.most_common())]

    intersection = set(valid_breakouts_feature_words).intersection(set(valid_no_breakouts_feature_words))
    print("intersection:\n", intersection)
    print("breakouts_unique:\n", set(valid_breakouts_feature_words)-intersection)
    print("no_breakouts_unique:\n", set(valid_no_breakouts_feature_words)-intersection)


def update_feature_words():
    conn = MyConn()
    df = pd.read_csv("../records/feature_words_diff.csv")
    valid_words = df[df["diff"]>=1]["feature_word"].values
    old_fw = [r[0].split() for r in conn.query(targets=["feature_words"], table="breakouts_feature_words_1")]
    new_fw = []
    for fw in old_fw:
        tmp = []
        for w in fw:
            if w in valid_words:
                tmp.append(w)
        # if len(tmp)>0:
        new_fw.append(tmp)

    print(len(old_fw), len(list(filter(None, new_fw))))
    for i in range(len(old_fw)):
        print(old_fw[i], new_fw[i])





def check_lyrics():
    breakout_tracks = open("../data/breakout_tracks_set_1.txt").read().splitlines()
    no_breakout_tracks = open("../data/no_breakout_tracks_set_1.txt").read().splitlines()
    conn = MyConn()
    d2v = Doc2Vec.load("/Users/inkding/Desktop/netease2/models/d2v/d2v_a1.mod")

    docs_vec = []
    docs_text = []
    size = 2000
    # tracks = breakout_tracks[:size] # 0.15985421647372214
    tracks = breakout_tracks[:size//2] + no_breakout_tracks[:size//2] # 0.16454543170333716
    # tracks = no_breakout_tracks[:size] # 0.18345906333057688

    for track_id in tracks:
        lyrics_path = conn.query(targets=["lyrics_path"], conditions={"track_id": track_id}, fetchall=False)[0]
        with open(lyrics_path) as f:
            content = json.load(f)
        text = content["lrc"]["lyric"]
        text = cut(replace_noise(text), join_en=False)
        docs_text.append(text)
        docs_vec.append(d2v.infer_vector(text))

    max_simi = 0
    match = {}
    # best_match = None
    for i in range(size-1):
        for j in range(i+1, size):
            match[(i,j)] = cosine_similarity(docs_vec[i], docs_vec[j])

    print(sum(match.values())/len(match))

    # sorted_match = sorted(list(match.items()), key=lambda p:p[1], reverse=True)

    # for i in range(50):
    #     print("{} - simi: {}".format(i, sorted_match[i][1]))
    #     print(docs_text[sorted_match[i][0][0]])
    #     print(docs_text[sorted_match[i][0][1]])
    #     print()




def rubbish_reviews_filter():
    conn = MyConn()
    words_counter = Counter()

    for res in conn.query(targets=["feature_words"], table="breakouts_feature_words_1"):
        feature_words = res[0].split()
        words_counter.update(feature_words)
    words_counter_size = sum(words_counter.values())
    words_counter_d = dict(map(lambda p:(p[0], p[1]/words_counter_size), words_counter.most_common()))

    special_words = open("../resources/special_words.txt").read().splitlines()

    output = []
    # 计算平均词频
    for res in conn.query(targets=["id","feature_words"], table="breakouts_feature_words_1"):
        beta, release_drive = conn.query(targets=["beta", "release_drive"], table="breakouts", conditions={"id": res[0]}, fetchall=False)
        if release_drive:
            continue
        feature_words = res[1].split()
        words = list(filter(lambda x:x not in special_words, feature_words))
        if len(words)<len(feature_words): 
            continue
        rubbish_score = np.mean([words_counter_d[w] for w in words])
        output.append((rubbish_score, res[0], beta, res[1]))

    output.sort(key=lambda x:x[0], reverse=True)
    for item in output:
        print(item)



def test_read_time():
    dir_ = "/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews_text/1/0"
    t = time.time()
    for file in os.listdir(dir_):
        if "DS" in file: continue
        # text = open(os.path.join(dir_, file)).read()[:10000]
        text = open(os.path.join(dir_, file)).read().splitlines()
        text = "\n".join(random.sample(text, min(2000, len(text))))
    print(time.time()-t)



if __name__ == '__main__':
    # check_feature_words()
    # check_lyrics()
    # conn = MyConn()
    # have_words = [r[0] for r in conn.query(table="no_breakouts_feature_words_1", targets=["id"])]
    # for id_ in have_words:
    #     conn.update(table="no_breakouts", settings={"have_words":1}, conditions={"id":id_})
    # update_feature_words()
    # rubbish_reviews_filter()
    test_read_time()


