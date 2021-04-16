import os
import json
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as mt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier

from connect_db import MyConn
from utils import *


conn = MyConn()
# sns.set_palette("RdYlBu")
sns.set_palette("GnBu_r")


def netease_tags_distribution():
    tags_counter = Counter()
    data = [r[0] for r in conn.query(sql="SELECT tags FROM tracks WHERE mp3_path is not null AND lyrics_path is not null")]
    data = [r[0] for r in conn.query(sql="SELECT tags FROM tracks WHERE mp3_path is not null AND lyrics_path is not null")]
    for item in data:
        tags = item.split()
        tags_counter.update(tags)
    print(sum(tags_counter.values()))
    l_tags_counts = list(tags_counter.most_common())
    others_count = 0
    for tag, count in l_tags_counts:
        if count<2000:
            others_count += count
    l_tags_counts.append(("others",others_count))
    with open("../thesis_related/netease_tags_distribution.txt", "w") as f:
        f.write("tag count\n")
        for tag, count in l_tags_counts:
            f.write("{} {}\n".format(tag, count))


def reviews_distribution():
    sns.set_style("white")
    data = conn.query(sql="SELECT reviews_num, first_review FROM tracks WHERE mp3_path is not null AND lyrics_path is not null")
    l_reviews_num, l_first_review = zip(*data)
    sns.displot(l_first_review)
    # sns.displot(l_reviews_num, color="orange",bins=list(range(1000, 30000, 1000)))
    plt.xlabel("date")
    plt.show()


def breakouts_distribution():
    sns.set_style("white")
    data = conn.query(sql="SELECT reviews_num, beta FROM breakouts")
    l_reviews_num, l_beta = zip(*data)
    # sns.distplot([x for x in l_beta if x<3000])
    sns.distplot([x for x in l_reviews_num if x<5000], color="orange")
    plt.xlabel("reviews_num")
    plt.show()


def reviews_num_plot():
    track_id = "115569"



def ML_comparison():
    # dataset_path = "../data/dataset_for_ML/mla1500_0316_10.pkl"
    dataset_path = "../data/dataset_for_ML/mla1500_0317_vggish.pkl"

    with open(dataset_path, 'rb') as f:
        X,y = pickle.load(f)
    # 标准化
    X = StandardScaler().fit_transform(X)
    # params for LightGBM
    params = {
        'learning_rate':0.1,
        'lambda_l1':0.1,
        'lambda_l2':0.2,
        'max_depth':6,
        "num_leaves": 50,
        'objective':'binary',
    }

    # models
    models = [
        LGBMClassifier(**params, random_state=21),
        AdaBoostClassifier(learning_rate=0.1, n_estimators=50, random_state=21),
        RandomForestClassifier(max_depth=10, n_estimators=30, random_state=21),
    ]
    CV = 5 #做五折交叉检验
    evaluations = []
    for model in models:
        model_name = model.__class__.__name__
        accs = cross_val_score(model, X, y, scoring="accuracy", cv=CV)
        for fold_id, acc in enumerate(accs):
            evaluations.append((model_name, fold_id, acc))
    cv_df = pd.DataFrame(evaluations,columns=['model', 'fold_id', 'accuracy'])


    print(cv_df.groupby('model').accuracy.mean())
        
    plt.figure(figsize=(8,6))
    sns.boxplot(x='model', y='accuracy', data=cv_df)
    sns.stripplot(x='model', y='accuracy', data=cv_df,
                 size=9, jitter=True, edgecolor='gray', linewidth=2)
    plt.show()


if __name__ == '__main__':
    # breakouts_distribution()
    ML_comparison()

