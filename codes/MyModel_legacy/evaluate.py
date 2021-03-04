import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec

from model import MusicFeatureExtractor, IntrinsicFeatureEmbed, SemanticFeatureEmbed, DNNClassifier
from config import Config
from data_loader import get_loader

def model1_process(models, batch_data):
    label, beta, mfcc, lyrics_vec, _ = batch_data
    batch_size = len(beta)
    # 直接将mfcc特征拉平
    flatten_mfcc = mfcc.reshape(batch_size, -1)
    # 将音频特征和歌词特征拼接在一起
    intrinsic_features = torch.cat((flatten_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 2w+)
    # 获得固有特征的嵌入表示（添加layernorm层）
    intrinsic_features = models["intrinsic_feature_embed"](intrinsic_features) # intrinsic_features: (batch_size, embedding_size)
    # 输入分类器
    predicted = models["dnn_classifier"](intrinsic_features)

    return predicted


def model24_process(models, batch_data):
    label, beta, mfcc, lyrics_vec, _ = batch_data
    batch_size = len(beta)
    # 使用CNN网络提取音频特征
    refined_mfcc = models["music_feature_extractor"](mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
    # 将音频特征和歌词特征拼接在一起
    intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
    # 获得固有特征的嵌入表示
    intrinsic_features = models["intrinsic_feature_embed"](intrinsic_features) # embed1: (batch_size, embedding_size)
    # 输入分类器
    predicted = models["dnn_classifier"](intrinsic_features)

    return predicted


def model3_process(models, batch_data, topk=6):
    '''
    model3不是端到端模型，只返回嵌入向量
    '''
    label, beta, mfcc, lyrics_vec, feature_words = batch_data
    batch_size = len(beta)
    feature_words = feature_words[:topk]

    feature_words = np.array(feature_words).transpose() # 将feature_words两个维度调换
    # 将关键词转换为词向量表示
    feature_words_vec = torch.tensor(list(map(word_to_vec, feature_words.ravel()))).reshape(-1, 300) # feature_words_vec: (batch_size*topk, 300)
    # 获取语义特征（关键词）的嵌入表示
    pos_embeds2 = models["semantic_feature_embed"](feature_words_vec).reshape(batch_size, topk, -1) # pos_embeds2: (batch_size, topk, embedding_size)
    # 使用CNN网络提取音频特征
    refined_mfcc = models["music_feature_extractor"](mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
    # 将音频特征和歌词特征拼接在一起
    intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
    # 获得固有特征的嵌入表示
    embed1 = models["intrinsic_feature_embed"](intrinsic_features) # embed1: (batch_size, embedding_size)

    return embed1



# def model5_process(models, batch_size, topk=6)
    

# 模型测试
def evaluate():
    # 加载模型
    models = {}
    config = Config()
    dataset, data_loader = get_loader(config, train=False)

    mf_path = "models/1/2/mf_extractor-e1.pkl"
    if_path = "models/1/2/if_embed-e1.pkl"
    dc_path = "models/1/2/dc_embed-e1.pkl"
    # music_feature_extractor = MusicFeatureExtractor(config)
    # music_feature_extractor.load_state_dict(torch.load(mf_path))
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config, in_dim=26140, embedding_size=512)
    # intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    intrinsic_feature_embed.load_state_dict(torch.load(if_path))
    dnn_classifier = DNNClassifier(config, embedding_size=512)
    # dnn_classifier = DNNClassifier(config)
    dnn_classifier.load_state_dict(torch.load(dc_path))

    models["intrinsic_feature_embed"] = intrinsic_feature_embed
    # models["music_feature_extractor"] = music_feature_extractor
    models["dnn_classifier"] = dnn_classifier

    for model_name in models:
        models[model_name].eval()

    # 将所有batch的结果放一起最后取平均
    corrects, loss = 0, 0
    predicted_all, label_all = np.array([]), np.array([])

    for batch_data in tqdm(data_loader, ncols=80):
        label, beta, mfcc, lyrics_vec, feature_words = batch_data
        # 选择测试模型
        predicted = model1_process(models, batch_data)

        corrects += (torch.max(predicted, 1)[1].view(label.size()).data == label.data).sum()
        loss += F.cross_entropy(predicted, label)
        predicted_all = np.append(predicted_all, torch.max(predicted, 1)[1].view(label.size()).data)
        label_all = np.append(label_all, label.data)


    size = len(data_loader.dataset)
    accuracy = corrects*1.0 / size
    avg_loss = loss / size 
    print("loss: {:.4f} acc: {:.4f}({}/{})\n".format(
                avg_loss, accuracy, corrects, size))
    print("confusion_matrix:\n", confusion_matrix(label_all, predicted_all))



if __name__ == '__main__':
    evaluate()

