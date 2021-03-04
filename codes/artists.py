import os
import re
import sys
import json
import pickle
import requests
import logging
import pandas as pd
import numpy as np
from gensim import models
from sklearn.preprocessing import StandardScaler

from connect_db import MyConn
from preprocess import cut


def cross_artists():
    '''
    考察知识库和网易云数据库的歌手交叉情况。
    '''
    def divide_artists():
        '''
        提取出爆发点和非爆发点涉及的歌手。
        save: "breakouts_artists.txt": 爆发点涉及的歌手
        save: "no_breakouts_artists.txt": 非爆发点涉及的歌手
        '''
        conn = MyConn()
        conditions = {"release_drive":0, "capital_drive":0, "fake":0}
        b_tracks = [r[0] for r in conn.query(targets=["distinct(track_id)"], table="breakouts", conditions=conditions)]
        nb_tracks = [r[0] for r in conn.query(targets=["distinct(track_id)"], table="no_breakouts")]

        b_arts, nb_arts = set(), set()
        for t in b_tracks:
            arts = conn.query(targets=["artist"], table="details", conditions={"track_id": t}, fetchall=False)[0].split(",")
            b_arts.update(arts)
        for t in nb_tracks:
            arts = conn.query(targets=["artist"], table="details", conditions={"track_id": t}, fetchall=False)[0].split(",")
            nb_arts.update(arts)

        with open("../data_related/breakouts_artists.txt", 'w') as f:
            f.write("\n".join(b_arts))
        with open("../data_related/no_breakouts_artists.txt", 'w') as f:
            f.write("\n".join(nb_arts))

    # divide_artists()
    b_arts0 = set(open("../data_related/breakouts_artists.txt").read().splitlines())
    nb_arts0 = set(open("../data_related/no_breakouts_artists.txt").read().splitlines())
    inter_arts = b_arts0.intersection(nb_arts0)
    b_arts, nb_arts = b_arts0 - inter_arts, nb_arts0 - inter_arts

    print("intersection num:{}, breakouts res: {}({}), no_breakouts res: {}({})".format(
                                    len(inter_arts), len(b_arts), len(b_arts0), len(nb_arts), len(nb_arts0)))

    with open("../data/name_2_KB_ents.pkl", "rb") as f:
        name_2_KB_ents = pickle.load(f)
    ents = set(name_2_KB_ents.keys())

    with open("../data/sup_artists_desc_2.json") as f:
        content = json.load(f)
    sup_arts = set([item["artist"].lower() for item in content])
    print("sup and ents:", len(sup_arts), len(sup_arts.intersection(ents)))

    b_arts = set(map(lambda x:x.lower().strip(), list(b_arts)))
    nb_arts = set(map(lambda x:x.lower().strip(), list(nb_arts)))
    inter_arts = set(map(lambda x:x.lower().strip(), list(inter_arts)))

    print("inter cross: {}, breakouts cross: {}, no_breakouts cross: {}".format(
            len(inter_arts.intersection(ents)), len(b_arts.intersection(ents)), len(nb_arts.intersection(ents))))

    print("ADD SUP:")
    print("inter cross: {}, breakouts cross: {}, no_breakouts cross: {}".format(
            len(inter_arts.intersection(ents.union(sup_arts))), len(b_arts.intersection(ents.union(sup_arts))), len(nb_arts.intersection(ents.union(sup_arts)))))


    # query_b_arts = b_arts - b_arts.intersection(ents)

def create_artists_table():
    '''
    在database中创建表格artists，包含id, nid, name。
    '''
    read_path = "/Volumes/nmusic/NetEase2020/data/simple_proxied_tracks_details"
    artists_set = set()
    conn = MyConn()

    for root, dirs, files in os.walk(read_path):
        for file in files:
            if "DS" in file: continue
            filepath = os.path.join(root, file)
            with open(filepath) as f:
                content = json.load(f)
            try:
                for ar in content["songs"][0]["ar"]:
                    artists_set.add((ar["id"], ar["name"]))
            except KeyboardInterrupt:
                print("interrupted by keyboard.")
                sys.exit(0)
            except Exception as e:
                print(filepath, e)


    print(len(artists_set))
    for ar in artists_set:
        conn.insert(table="artists", settings={"nid":ar[0], "name":ar[1]})


def add_desc_source_2_database():
    '''
    将歌手的文本来源写入数据库表格。
    '''
    conn = MyConn()
    with open("../data/name_2_KB_ents.pkl", "rb") as f:
        name_2_KB_ents = pickle.load(f)
    ents = set(name_2_KB_ents.keys())

    with open("../data/sup_artists_desc_2.json") as f:
        content = json.load(f)
    sup_arts = set([item["artist"].lower() for item in content])
    # print("KB: {}, Netease: {}, intersection: {}".format(len(ents), len(sup_arts), len(sup_arts.intersection(ents))))

    for key in ents:
        conn.update(table="artists", settings={"source":1}, conditions={"lower_name":key})
    for name in sup_arts:
        conn.update(table="artists", settings={"source":2}, conditions={"lower_name":name})



def get_description_by_api():
    conn = MyConn()
    res = conn.query(targets=["name", "nid"], table="artists")
    name_2_id = dict([(r[0].lower().strip(), r[1]) for r in res])
    artists = open("../data_related/query_artists.txt").read().splitlines()
    print(len(artists))
    

    url_base = 'http://127.0.0.1:3000'
    # 代理服务器
    proxyHost = "http-dyn.abuyun.com"
    proxyPort = "9020"

    # 代理隧道验证信息
    proxyUser = "H941185H9V92U03D"
    proxyPass = "0B5117D9D8FABBD1"

    proxy = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
      "host" : proxyHost,
      "port" : proxyPort,
      "user" : proxyUser,
      "pass" : proxyPass,
    }

    data = []
    for ar in artists:
        try:
            id_ = name_2_id[ar]
            if id_=="0":
                continue
            url = url_base + "/artist/desc?id={}&proxy={}".format(id_, proxy)
            res = requests.get(url, timeout=10).json()
            res["id"] = id_
            res["artist"] = ar
            data.append(res)
        except KeyboardInterrupt:
            print("interrupted by keyboard.")
            sys.exit(0)
        except Exception as e:
            print(ar, e)

    with open("../data/sup_artists_desc.json", 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        



def artists_KB():
    '''
    将知识库中和artists相关的实体信息提取出来。
    save: "KB_artists.txt": 存在于知识库中的所有歌手实体
    save: "artists_KB.csv": 与歌手相关的所有知识库信息
    save: "name_2_KB_ents.pkl": 实体与原型的对应关系（如("五月天", "五月天[摇滚乐队]")）
    '''
    df0 = pd.read_csv("/Users/inkding/data/ownthink_v2.csv")
    artists_tags = open("../data_related/artists_tags.txt").read().splitlines()

    df0["is_artist"] = df0["值"].map(lambda x:1 if x in artists_tags else 0)
    df1 = df0[df0["is_artist"]==1]
    df1 = df1[(df1["属性"]=="标签") | (df1["属性"]=="职业")] 

    with open("../data/KB_artists.txt", "w") as f: 
        f.write("\n".join(list(df1.unique())))

    KB_artists = set(KB_artists) # 变为set可以大大加速in查询（哈希底层）
    df0["is_artist_ent"] = df0["实体"].map(lambda x:1 if x in KB_artists else 0)  
    new_df = df0[df0["is_artist_ent"]==1]
    new_df.drop(["is_artist", "is_artist_ent"], axis=1)

    new_df.to_csv("../data/artists_KB.csv", index=False, encoding="utf-8-sig")

    # 实体与原型的对应关系
    new_df = pd.read_csv("../data/artists_KB.csv")
    KB_ents = list(new_df["实体"].unique())
    # 原型
    KB_ents_ori = list(map(lambda x:re.sub(r"\[.+\]", "", x.lower().strip()), KB_ents))
    name_2_KB_ents = {}

    for i in range(len(KB_ents)):
        k, v = KB_ents_ori[i], KB_ents[i]
        if "《" in k: continue
        if k in name_2_KB_ents:
            name_2_KB_ents[k].append(v)
        else:
            name_2_KB_ents[k] = [v]
    with open("../data/name_2_KB_ents.pkl", "wb") as f:
        pickle.dump(name_2_KB_ents, f)




def generate_description_KB(artist, KB_df, name_2_KB_ents, mode=1):
    '''
    已知歌手姓名，基于知识库生成描述文本。
    param: KB_df[pandas.DataFrame]: 知识库
    param: name_2_KB_ents[dict]: 歌手在知识库中的对应实体字典
    param: mode[int]: 1 - 只使用"描述"属性；2 - 使用"描述"和其他属性信息；3 - 使用"描述"和其他属性信息，且保留关键词
    return: text[str]
    '''
    data_field = ["描述", "别名", "擅长曲风", "音乐类型", "团队成员", "职业", "经纪公司", "主要成就", "代表作品", "好友", "搭档", "队友"]

    if mode==1:
        for ent in name_2_KB_ents[artist]:
            ent_df = KB_df[KB_df["实体"]==ent]
            for index, item in ent_df.iterrows():
                if item["属性"]=="描述":
                    return item["值"]

    else: # mode = 2 or 3
        data_d = {}
        for attr in data_field:
            data_d[attr] = set()
        for ent in name_2_KB_ents[artist]:
            ent_df = KB_df[KB_df["实体"]==ent]
            for index, item in ent_df.iterrows():
                attr, value = item["属性"], item["值"]
                if attr in ["描述", "经纪公司"] and len(data_d[attr])==1: continue
                if attr in data_d and len(data_d[attr])<3:
                    data_d[attr].add(str(value))

        for attr in data_d:
            data_d[attr] = " ".join(data_d[attr])

        text = ""
        if mode==2:
            text = " ".join(list(data_d.values()))
        else:
            # 保留关键词（属性名称）
            for k, v in data_d.items():
                if len(v)>0:
                    text += "{}: {} ".format(k, v)

        return text
    

def generate_description_sup(artist, sup_data):
    '''
    利用网易云提供的补充信息生成歌手描述文本
    param: sup_data[list(dict)]: 补充信息
    param: artist: 歌手姓名
    return: text: 描述文本
    '''
    for item in sup_data:
        if item["artist"].lower().strip() == artist.lower().strip():
            text = ""
            for intro in item["intro"]:
                text += intro["txt"]
                if len(text)>=300: break
            return text[:300]



class ArtistsDoc2VecGenerator:
    '''
    准备 Doc2Vec 模型的数据输入。定义生成器。
    使用全部的语料。
    '''
    def __init__(self, artists_documents_save_path=None, mode=1):
        '''
        param: artists_documents_save_path: 如果不为None，则将描述文本保存
        '''
        self.artists_documents_save_path = artists_documents_save_path
        self.KB_df = pd.read_csv("../data/artists_KB.csv")
        with open("../data/name_2_KB_ents.pkl", "rb") as f:
            self.name_2_KB_ents = pickle.load(f)
        with open("../data/sup_artists_desc_2.json") as f:
            self.sup_data = json.load(f)
        self.mode = mode # 生成艺人相关文本时的模式（KB源）
        self.stops_sup = ["日出", "最佳", "冠军", "亚军", "季军", "参加", "获得"]

        # 使用KB和sup_data中的全部数据
        self.artists = set([(ar.lower().strip(), "KB") for ar in self.name_2_KB_ents])
        for item in self.sup_data:
            if item["artist"].lower().strip() not in self.artists:
                self.artists.add((item["artist"].lower().strip(),"sup"))
        print("train size: ", len(self.artists))


    def __iter__(self):
        self.artists_documents = []
        self.artists_flag = 0

        for ar, source in self.artists:
            ar = ar.lower().strip()
            if source=="KB":
                text = str(generate_description_KB(ar, self.KB_df, self.name_2_KB_ents, self.mode))
            elif source=="sup":
                text = str(generate_description_sup(ar, self.sup_data))
            words = cut(text, stops_sup=self.stops_sup, filter_number=True)
            self.artists_documents.append((ar, source, text))
            self.artists_flag += 1

            yield models.doc2vec.TaggedDocument(words, [str(self.artists_flag)])


    def save_artists_documents(self):
        '''
        将flag与文本信息对应，保存没有实体对应的歌手名称。
        save: "artists_d2v_docs_{}.json": 歌手flag与描述信息
        save: "nan_artists.txt": 没有实体对应的歌手
        '''
        data = [{"flag": index, "artist": item[0], "source": item[1], "desc": item[2]} 
                        for index, item in enumerate(self.artists_documents)]

        print("saving {} artists documents ...".format(self.artists_flag))
        if self.artists_documents_save_path:
            with open(self.artists_documents_save_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)



def train_artists_doc2vec(mode, window):
    '''
    训练 Doc2Vec 模型。
    '''
    artists_documents_save_path = "../data/artists_desc_m{}_sup.json".format(mode)
    # for item in ArtistsDoc2VecGenerator(artists_documents_save_path, mode=mode):
    #     print(item.tags, item.words)
    
    print("mode: {}, window_size: {}".format(mode, window))
    logging.basicConfig(level=logging.INFO, filename="../logs/artists_d2v_2.log",
                        format="%(asctime)s : %(message)s", datefmt="%H:%M:%S")
    generator = ArtistsDoc2VecGenerator(artists_documents_save_path, mode=mode)
    model = models.Doc2Vec(documents=generator, dm=1, vector_size=300, window=window, 
                    workers=6, epochs=5)
    generator.save_artists_documents()
    model.save("../models/d2v/ar_m{}w{}_sup.mod".format(mode, window))



class ArtistsDoc2vecTest:
    def __init__(self, model_path, model_docs_path):
        self.model = models.Doc2Vec.load(model_path)
        with open(model_docs_path) as f:
            self.docs = json.load(f)    

    def most_similar_within_model(self, text, topn=10):
        words = cut(text)
        vec = self.model.infer_vector(words)
        s = self.model.docvecs.most_similar([vec], topn=topn)
        for index, score in s:
            print(index, score, self.docs[int(index)]["artist"])
            print(self.docs[int(index)]["desc"], "\n")

    def count_similarity(self, text1, text2):
        words1 = cut(text1)
        words2 = cut(text2)
        print("simi score: {:.3f}".format(
            model.docvecs.similarity_unseen_docs(words1, words2)))


# 之前的都是笑话哈哈哈哈哈哈哈哈哈哈哈

def artist_vec_from_tags(min_tags_num=2):
    conn = MyConn()
    artists = conn.query(table="artists", targets=["name", "nid"])
    tracks_artists = conn.query(table="details", targets=["track_id", "artists"])
    d_artist_tracks = {}
    for ar, nid in artists:
        if nid=="0": continue
        d_artist_tracks[ar.lower().strip()] = []

    tracks = set()
    for tid, t_artists in tracks_artists:
        tracks.add(tid)
        t_artists = t_artists.lower().strip().split(",")
        for ar in t_artists:
            if ar in d_artist_tracks:
                d_artist_tracks[ar].append(tid)


    tracks_tags = conn.query(sql="SELECT track_id, tags FROM tracks")
    tags = open("../data_related/自带tags.txt").read().splitlines()
    d_tag_index = dict([(t, i) for i, t in enumerate(tags)])
    d_track_tags_count = {}
    for tid, t_tags in tracks_tags:
        if tid not in tracks: continue
        t_vec = np.zeros((len(tags),))
        t_tags = t_tags.split()
        for t in t_tags:
            t_vec[d_tag_index[t]] += 1
        d_track_tags_count[tid] = t_vec

    d_artist_tags_count = {}
    for ar, ar_tracks in d_artist_tracks.items():
        if len(ar_tracks)==0: continue
        ar_vec = np.sum(np.array([d_track_tags_count[tid] for tid in ar_tracks]), axis=0)
        if np.sum(ar_vec, axis=None)>=min_tags_num:
            d_artist_tags_count[ar] = ar_vec

    artists = list(d_artist_tags_count.keys())
    ar_vecs = list(d_artist_tags_count.values())
    norm_ar_vecs = StandardScaler().fit_transform(ar_vecs)

    # 统计
    tags_count = np.sum(np.array(ar_vecs), axis=0)
    # for i in range(len(tags)):
    #     print(tags[i], tags_count[i])
    print(len(artists))

    d_artist_vec = dict(zip(artists, norm_ar_vecs))
    with open("../data/artists_vec_dict.pkl", "wb") as f:
        pickle.dump(d_artist_vec, f)



if __name__ == '__main__':
    # func_name = sys.argv[1]
    # if input("是否输入参数？(y/n)")=="y":
    #     params = list(map(int, sys.argv[2:]))
    # else:
    #     params = []
    # print("[file]: {}, [function]: {}, [params]: {}\n".format(__file__, func_name, params), "="*20)

    # if params:
    #     eval(func_name)(*params)
    # else:
    #     eval(func_name)()
    artist_vec_from_tags()


