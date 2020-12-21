import os
import json
import pandas as pd
import numpy as np
import datetime

from sklearn.cluster import KMeans
from sklearn import metrics

from connect_db import MyConn
from breakout_tools import *

def get_breakouts_json():
    '''
    最开始一版的数据提取出来保存在json文件中，后来才转移进数据库。
    '''
    path = "/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews"
    json_path = "../data/breakouts-0.json"
    w_prefix = "/Volumes/nmusic/NetEase2020/data/breakouts_text/breakouts-0"
    n_dir = 2
    dir_size = (100, 100)

    # 把没有lyrics和mp3的去除
    conn = MyConn()
    res = conn.query(targets=["track_id"], conditions={"have_lyrics":1, "have_mp3":1})
    backup_tracks = set()
    for r in res:
        backup_tracks.add(str(r[0]))

    write_content = []
    total_count = 0
    for root,dirs,files in os.walk(path):
        for file in files:
            n = int(root.split('/')[-2])*100 + int(root.split('/')[-1])
            if "DS" in file or file[:-5] not in backup_tracks:
                continue
            try:
                track_id = file[:-5]
                filepath = os.path.join(root, file)
                df = get_reviews_df(filepath)
                reviews_count, dates =  get_reviews_count(df["date"].values)

                breakouts_group = get_breakouts(reviews_count, min_reviews=200)
                breakouts = [g[0] for g in breakouts_group]
                bdates = [dates[p[0]] for p in breakouts]

                for i, p in enumerate(breakouts):
                    beta = p[1]
                    date = bdates[i]
                    reviews_num = reviews_count[p[0]]
                    btext = get_breakouts_text(df, date)
                    w_path = os.path.join(
                        assign_dir(prefix=w_prefix, n_dir=n_dir, dir_size=dir_size, flag=total_count),
                        "{}-{}.txt".format(track_id, i)
                    )
                    # with open(w_path, 'w') as f:
                    #   f.write(btext)
                    write_content.append({
                        "track_id": track_id,
                        "flag": i,
                        "beta": beta,
                        "date": date,
                        "reviews_num": reviews_num,
                        "text_path": w_path
                    })
                    total_count += 1
                    if total_count % 100 == 0:
                        print("total_count = {}".format(total_count))
                        if total_count==200:
                            with open("../data/breakouts-00.json", 'w') as f:
                                json.dump(write_content, f, ensure_ascii=False, indent=2)
            except:
                print(traceback.format_exc())
                return

    with open(json_path, 'w') as f:
        json.dump(write_content, f, ensure_ascii=False, indent=2)




def prep_no_breakouts_data():
    '''
    description:
        + 提取no_breakouts_points并存入数据库
            - 每首歌曲至多抽取3个时间点（不满3个则是用全部）
        + 将每个point对应的评论文本提取并保存为txt文件
    params:
        + r_path, w_path: 读写路径
    '''

    conn = MyConn()
    r_path = "/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews/"
    breakouts_tracks = set([r[0] for r in conn.query(table="breakouts", targets=["track_id"])])
    # 保存路径设置
    count = 0
    prefix = "/Volumes/nmusic/NetEase2020/data/no_breakouts_text"
    n_dir = 2
    dir_size = (100, 100)

    for root, dirs, files in os.walk(r_path):
        for file in files:
            try:
                if "DS" in file: continue
                filepath = os.path.join(root, file)
                track_id = file[:-5]
                if track_id in breakouts_tracks: 
                    continue

                if len(conn.query(table="no_breakouts", targets=["track_id"], conditions={"track_id": track_id})) != 0:
                    continue

                df = get_reviews_df(filepath)
                reviews_count, dates = get_reviews_count(df["date"].values)

                no_breakouts_group = get_no_breakouts(reviews_count, min_reviews=200, thres=10)
                # 抽样
                samples = np.floor(np.linspace(0, len(no_breakouts_group)-1, min(len(no_breakouts_group), 3)))
                no_breakouts_group = [no_breakouts_group[int(s)] for s in samples]

                if len(no_breakouts_group)>0:
                    # print(file, len(no_breakouts_group))
                    for flag, group in enumerate(no_breakouts_group):
                        # 基本信息上传至数据库
                        # group: (left_index, right_index, reviews_acc)
                        data = {
                            "id": '-'.join([track_id, str(flag)]),
                            "track_id": track_id,
                            "flag": flag,
                            "start_date": dates[group[0]],
                            "end_date": dates[group[1]],
                            "reviews_acc": group[2]
                        }
                        conn.insert(table="no_breakouts", settings=data)

                        # 提取文字
                        text = ""
                        dir_ = assign_dir(prefix, n_dir, dir_size, flag=count)
                        for point in range(group[0], group[1]):
                            date = dates[point]
                            text += get_breakouts_text(df, date)
                        with open(os.path.join(dir_, "{}-{}.txt".format(file[:-5], flag)), 'w') as f:
                            f.write(text)

                        count += 1
                        if count%100==0:
                            print(count)

            except KeyboardInterrupt:
                print("interrupted by keyboard.")
                return
            except:
                print(traceback.format_exc())
                continue



def breakouts_curve():
    '''
    绘制爆发曲线
    '''
    conn = MyConn()

    for i in range(6):
        save_dir = "../data/breakouts_curve_clusters/{}".format(i)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tracks = [r[0] for r in conn.query(targets=["track_id"], table="breakouts_complements", conditions={"label6":i})]
        for tid in tracks[:100]:
            save_path = os.path.join(save_dir, "{}.png".format(tid))
            view_reviews_num_curve(tid, save_path=save_path)



def identify_release_drive_breakouts():
    '''
    找出由于新歌发布导致爆发的样本点（位于最头部）
    '''
    conn = MyConn()
    breakouts = conn.query(targets=["id", "track_id", "date"], table="breakouts")

    release_breakouts_count = 0
    release_breakouts_tracks_set = set()
    more_breakouts_tracks_set = set()
    for b in breakouts:
        track_first_review = conn.query(targets=["first_review"], conditions={"track_id":b[1]}, fetchall=False)[0]
        if b[2] - track_first_review < datetime.timedelta(days=15):
            release_breakouts_count += 1
            release_breakouts_tracks_set.add(b[1])
            conn.update(table="breakouts", settings={"release_drive":1}, conditions={"id":b[0]})
        else:
            more_breakouts_tracks_set.add(b[1])

    print(release_breakouts_count)
    print(len(release_breakouts_tracks_set))
    print(len(more_breakouts_tracks_set))




def breakouts_complements():
    '''
    对爆发信息进一步补充（用于对爆发分类）
    '''
    conn = MyConn()
    logspace = [(0,100), (100,180), (180,326), (326,589), (589,1066), (1066,3494), (3494,30000)]
    blevel_num = len(logspace)
    logspace_count = dict(zip(logspace, blevel_num*[0]))
    breakout_tracks = [r[0] for r in conn.query(targets=["DISTINCT(track_id)"], table="breakouts",
                                                conditions={"release_drive":0})]

    for track_id in breakout_tracks:
        reviews_num, first_review, last_review = conn.query(targets=["reviews_num","first_review","last_review"], 
                                    conditions={"track_id":track_id}, fetchall=False)
        breakouts = conn.query(targets=["flag","reviews_num","beta","release_drive"], table="breakouts", conditions={"track_id":track_id})
        days_num = (last_review - first_review).days
        # 除去爆发点的平均评论数
        avg_normal = float((reviews_num - np.sum([b[1] for b in breakouts])) / (days_num - len(breakouts)))

        blevel_vec = blevel_num*[0]
        for b in breakouts:
            if b[3]==1: continue # 不考虑release_drive爆发
            for i in range(blevel_num):
                if b[2]>=logspace[i][0] and b[2]<logspace[i][1]: # 考察beta区间
                    blevel_vec[i] += 1
                    logspace_count[logspace[i]] += 1
                    break

        breakouts_num = int(np.sum(blevel_vec))
        blevel = 0
        for i in range(len(blevel_vec)):
            blevel += i*blevel_vec[i]
        blevel = blevel*1.0 / breakouts_num
        settings = {
            "track_id": track_id,
            "average_reviews_num": avg_normal,
            "blevel_vec": ' '.join(map(str,blevel_vec)),
            "breakouts_num": breakouts_num,
            "blevel": blevel
        }
        conn.insert_or_update(table="breakouts_complements", settings=settings)
        # print(settings)
        print(track_id)


def breakouts_complements_cluster():
    '''
    利用breakouts_complements中average_reviews_num, breakouts_num, blevel / blevel_vec信息作聚类。
    测试多种聚类数目。
    并在breakouts_complements中更新track的聚类label。
    '''

    n_clusters = [5,6,7,8,9,10,11,12,13,14,15]
    conn = MyConn()
    # res = conn.query(table="breakouts_complements", targets=["track_id","average_reviews_num","breakouts_num","blevel"])
    res = conn.query(table="breakout_tracks_complements", targets=["track_id","average_reviews_num","breakouts_num"])
    # res = conn.query(table="breakouts_complements", targets=["track_id","breakouts_num"])
    tracks = [r[0] for r in res]
    data = [r[1:] for r in res]

    def test_n_clusters():
        for n in n_clusters:
            model = KMeans(n_clusters=n, random_state=21)
            labels = model.fit_predict(data)
            n_labels = len(set(labels))
            print("n_clusters:", n)
            print("inertia:", model.inertia_)
            print("silhouette_score:", metrics.silhouette_score(data, labels))

            res = []
            for i in range(n_labels):
                child_data = [data[j] for j in range(len(data)) if labels[j]==i]
                # print("mean-average_reviews_num: {:.3f}, size: {}".format(np.mean(list(zip(*child_data))[0]), len(child_data)))
                res.append([np.mean(list(zip(*child_data))[0]), np.mean(list(zip(*child_data))[1]), len(child_data)])
            res = sorted(res, key=lambda x:x[0])
            for r in res:
                print("mean-average_reviews_num: {:.3f}, mean-breakouts_num: {:.3f}, size: {}".format(
                    r[0], r[1], r[2]))
            print()

    def assign_cluster(n_clusters=6):
        model = KMeans(n_clusters=n_clusters, random_state=21)
        labels = model.fit_predict(data)
        for i in range(len(tracks)):
            track_id = tracks[i]
            label = labels[i]
            conn.update(table="breakout_tracks_complements", settings={"label":int(label)}, conditions={"track_id":track_id})

    def cluster_special_words():
        cluster_2_pairs = {}
        sql = "SELECT track_id, label6, special_words FROM breakout_tracks_complements WHERE special_words IS NOT NULL and length(special_words)>0" 
        # for track_id, label, special_words in conn.query(sql=sql):
        #     if label in cluster_2_pairs:
        #         cluster_2_pairs[label].append((track_id, special_words))
        #     else:
        #         cluster_2_pairs[label] = [(track_id, special_words)]
        # for k, v in cluster_2_pairs.items():
        #     print(k)
        #     for p in v:
        #         print(p[0], p[1])
        #     print()
        for track_id, label, special_words in conn.query(sql=sql):
            if label in cluster_2_pairs:
                cluster_2_pairs[label].extend(list(set(special_words.split())))
            else:
                cluster_2_pairs[label] = list(set(special_words.split()))

        for k, v in cluster_2_pairs.items():
            counter = Counter(v)
            print(k)
            for i, p in enumerate(counter.most_common()):
                print("{} {} {:.3f}".format(i, p[0], p[1]/sum(counter.values())))
            print()


    # test_n_clusters()
    # assign_cluster()
    cluster_special_words()



def breakouts_curve_with_special_words():
    conn = MyConn()        
    # special_words = ["高考", "节日", "微博", "抖音", "b站", "bgm", "翻唱", "rip"]
    special_words = ["b站", "抖音", "bgm"]
    special_word_2_tracks = {}
    for w in special_words:
        special_word_2_tracks[w] = set()

    for id_, text in conn.query(targets=["id", "special_words"], table="breakouts_feature_words_c3"):
        track_id = id_.split('-')[0]
        for w in text.split():
            if w in special_words:
                special_word_2_tracks[w].add(track_id)

    for k, v in special_word_2_tracks.items():
        print(k, len(v))
        save_dir = "../data/breakouts_curve/special_words_c3/{}".format(k)
        for track_id in v:
            view_reviews_num_curve_html(track_id, save_dir=save_dir)



def get_breakouts_num():
    conn = MyConn()
    breakouts = conn.query(targets=["id", "track_id"], table="breakouts")
    track_2_bnum = {}
    for id_, track_id in breakouts:
        if track_id in track_2_bnum:
            track_2_bnum[track_id] += 1
        else:
            track_2_bnum[track_id] = 1
    for k, v in track_2_bnum.items():
        conn.update(table="sub_tracks", settings={"bnum": v}, conditions={"track_id": k})





if __name__ == '__main__':
    breakouts_complements_cluster()
    # breakouts_curve_with_special_words()
    # get_breakouts_num()

