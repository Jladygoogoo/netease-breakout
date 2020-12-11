import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pairwise_embed(feature_words, embed1, semantic_feature_embed, word_to_vec):
    '''
    负样本抽取，并转变为嵌入表示。
    params: 
        feature_words: ndarray: (batch_size, topk,)
        embed1: ndarray: (batch_size, embedding_size)
        semantic_feature_embed: 语义特征嵌入模型
        word_to_vec: 函数，获取词向量
    return: 
        pairwise_embed: ndarray: (batch_size, topk, embedding_size)
    '''
    batch_size = len(feature_words)
    topk = len(feature_words[0])
    candidates = set(np.array(feature_words).ravel()) # 整个batch的关键词词库
    # 每一首歌的pos_feature_words和neg_feature_words
    pos_neg_words_l = []

    def dist(word, embed1, semantic_feature_embed, word_to_vec):
        input_ = torch.tensor(word_to_vec(word)).unsqueeze(0)
        output_ = semantic_feature_embed(input_).squeeze()
        
        return np.linalg.norm(embed1.detach().numpy() - output_.detach().numpy())

    pairwise_words_vec = []
    with torch.no_grad():
        for i in range(batch_size):
            # 排除pos_feature_words
            valid_candidates = candidates - set(feature_words[i]) 
            # 从所有关键词中随机一小批用于距离排序
            hits = np.random.choice(list(valid_candidates), size=50, replace=False)
            hits = sorted(hits, key=lambda w:dist(w, embed1[i], semantic_feature_embed, word_to_vec))[:topk]
            pos_neg_words_l.append((list(feature_words[i]), hits))
            pairwise_words_vec.append(list(map(word_to_vec, hits)))

    pairwise_words_vec = np.array(pairwise_words_vec).reshape(-1, 300)
    pairwise_embed = semantic_feature_embed(torch.tensor(pairwise_words_vec)).reshape(batch_size, topk, -1)

    return pairwise_embed, pos_neg_words_l


def beta_transform(beta):
    '''
    基于爆发因子分配权重，爆发因子越大权重越大。
    params: beta: 爆发因子
    return: weight: ~(1, 10)
    '''
    if beta<=225:
        return 1 + 9 / (1 + (1+0.05)**(225-beta))
    return 1 + 9 / (1 + (1+0.05)**(-(beta-225)**0.5))


def my_loss1(embed1, pos_embeds2):
    '''
    不使用beta信息和负采样。向量距离采用欧式距离。
    params: embed1: (batch_size, embedding_size)
    params: pos_embeds2: (batch_size, topk, embedding_size)
    return: loss: 标量值，带梯度
    '''
    batch_size, topk = len(pos_embeds2), len(pos_embeds2[0])
    embed1 = embed1.unsqueeze(1).expand(batch_size, topk, embedding_size) # embed1: (batch_size, topk, embedding_size)

    loss = torch.mean(F.pairwise_distance(embed1.reshape(batch_size*topk, -1), pos_embeds2.reshape(batch_size*topk, -1)))
    return loss

def my_loss2(embed1, pos_embeds2, beta):
    '''
    使用beta信息，不使用负采样。向量距离采用欧式距离。
    params: 
        embed1: (batch_size, embedding_size)
        pos_embeds2: (batch_size, topk, embedding_size)
        beta: (batch_size,)
    return: 
        loss: 标量值，带梯度
    '''
    batch_size, topk = len(pos_embeds2), len(pos_embeds2[0])
    embed1 = embed1.unsqueeze(1).expand(batch_size, topk, -1) # embed1: (batch_size, topk, embedding_size)
    beta = beta.unsqueeze(1).expand(batch_size, topk).reshape(batch_size*topk) # beta: (batch_size*topk, )
    pos_weight = torch.tensor(np.vectorize(beta_transform)(beta))

    pos_dists = F.pairwise_distance(embed1.reshape(batch_size*topk, -1), pos_embeds2.reshape(batch_size*topk, -1))
    loss = torch.mean(pos_dists.mul(pos_weight))

    return loss

def my_loss3(embed1, pos_embeds2, neg_embeds2, beta):
    '''
    使用beta信息和负采样。向量距离采用欧式距离。
    params: 
        embed1: (batch_size, embedding_size)
        pos_embeds2: (batch_size, topk, embedding_size)
        neg_embeds2: (batch_size, topk, embedding_size)
        beta: (batch_size,)
    return: 
        loss: 标量值，带梯度
    ''' 
    batch_size, topk = len(pos_embeds2), len(pos_embeds2[0])
    embed1 = embed1.unsqueeze(1).expand(batch_size, topk, -1) # embed1: (batch_size, topk, embedding_size)
    beta = beta.unsqueeze(1).expand(batch_size, topk).reshape(batch_size*topk) # beta: (batch_size*topk,)
    pos_weight = torch.tensor(np.vectorize(beta_transform)(beta))

    pos_dists = F.pairwise_distance(embed1.reshape(batch_size*topk, -1), pos_embeds2.reshape(batch_size*topk, -1))
    neg_dists = F.pairwise_distance(embed1.reshape(batch_size*topk, -1), neg_embeds2.reshape(batch_size*topk, -1))
    
    loss = torch.mean(pos_dists.mul(pos_weight)) - torch.mean(neg_dists)

    return loss







