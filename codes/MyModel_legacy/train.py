import os
import sys
import pickle
import numpy as np 
from tqdm import tqdm
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_loader
from config import Config
from model import MusicFeatureExtractor, IntrinsicFeatureEmbed, SemanticFeatureEmbed, DNNClassifier
from model_utils import get_pairwise_embed, my_loss1, my_loss2, my_loss3

# device configuratuon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configurations
config = Config()
w2v_model = Word2Vec.load(config.w2v_path)
def word_to_vec(word): # 得到词语的向量表示
    return w2v_model.wv[word] # 默认w2v_model.wv.__contains__(word)=True


# none + 端到端
def model1(data_loader):
    # models
    dnn_classifier = DNNClassifier(config, embedding_size=512)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config, in_dim=26140, embedding_size=512)

    # loss and optimizer
    params = list(dnn_classifier.parameters()) + list(intrinsic_feature_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/1/{}".format(model_no)):
        os.makedirs("models/1/{}".format(model_no))

    # 训练模型
    losses = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for label, beta, mfcc, lyrics_vec, _ in tqdm(data_loader, ncols=80):
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)
            batch_size = len(beta)

            # 直接将mfcc特征拉平
            flatten_mfcc = mfcc.reshape(batch_size, -1)
            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((flatten_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 2w+)
            # 获得固有特征的嵌入表示（添加layernorm层）
            intrinsic_features = intrinsic_feature_embed(intrinsic_features) # intrinsic_features: (batch_size, embedding_size)
            # 输入分类器
            predict = dnn_classifier(intrinsic_features)

            # 计算损失
            loss = F.cross_entropy(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss.item()))
                losses["{}-{}".format(epoch, step)] = loss.item()


        # save models
        if epoch % config.save_epoch == 0:
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/1/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(dnn_classifier.state_dict(), 
                        "models/1/{}/dc_embed-e{}.pkl".format(model_no, epoch))
            with open("models/1/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)


# cnn + 端到端
def model2(data_loader):
    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    dnn_classifier = DNNClassifier(config)

    # loss and optimizer
    params = list(music_feature_extractor.parameters()) + list(intrinsic_feature_embed.parameters()) + list(dnn_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/2/{}".format(model_no)):
        os.makedirs("models/2/{}".format(model_no))

    # 训练模型
    losses = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for label, beta, mfcc, lyrics_vec, _ in tqdm(data_loader, ncols=80):
            # beta: (batch_size,)
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)

            batch_size = len(beta)

            # 使用CNN网络提取音频特征
            refined_mfcc = music_feature_extractor(mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
            # 获得固有特征的嵌入表示
            intrinsic_features = intrinsic_feature_embed(intrinsic_features) # embed1: (batch_size, embedding_size)
            # 输入分类器
            predict = dnn_classifier(intrinsic_features)

            # 计算损失
            loss = F.cross_entropy(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss.item()))
                losses["{}-{}".format(epoch, step)] = loss.item()

        # save models
        if epoch % config.save_epoch == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "models/2/{}/mf_extractor-e{}.pkl".format(model_no, epoch))
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/2/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(dnn_classifier.state_dict(), 
                        "models/2/{}/dc_embed-e{}.pkl".format(model_no, epoch))
            with open("models/2/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)



# cnn + 评论数据
def model3(data_loader):
    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    semantic_feature_embed = SemanticFeatureEmbed(config)

    # loss and optimizer
    params = list(music_feature_extractor.parameters()) + list(intrinsic_feature_embed.parameters()) + list(semantic_feature_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/3/{}".format(model_no)):
        os.makedirs("models/3/{}".format(model_no))

    # 训练模型
    losses = {}
    pos_neg_words = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for label, beta, mfcc, lyrics_vec, feature_words in tqdm(data_loader, ncols=80):
            # beta: (batch_size,)
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)
            # feature_words: (topk, batch_size) => 注意第一维不是batch_size

            batch_size = len(beta)
            topk = config.topk
            feature_words = feature_words[:topk]

            feature_words = np.array(feature_words).transpose() # 将feature_words两个维度调换
            # 将关键词转换为词向量表示
            feature_words_vec = torch.tensor(list(map(word_to_vec, feature_words.ravel()))).reshape(-1, 300) # feature_words_vec: (batch_size*topk, 300)
            # 获取语义特征（关键词）的嵌入表示
            pos_embeds2 = semantic_feature_embed(feature_words_vec).reshape(batch_size, topk, -1) # pos_embeds2: (batch_size, topk, embedding_size)
            # 使用CNN网络提取音频特征
            refined_mfcc = music_feature_extractor(mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
            # 获得固有特征的嵌入表示
            embed1 = intrinsic_feature_embed(intrinsic_features) # embed1: (batch_size, embedding_size)

            # 计算损失
            loss = my_loss2(embed1, pos_embeds2, beta)

            # zero_grad将上一次梯度更新的结果扔掉
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss.item()))
                losses["{}-{}".format(epoch, step)] = loss.item()

        # save models
        if epoch % config.save_epoch == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "models/3/{}/mf_extractor-e{}.pkl".format(model_no, epoch))
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/3/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(semantic_feature_embed.state_dict(), 
                        "models/3/{}/sf_embed-e{}.pkl".format(model_no, epoch))
            with open("models/3/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)




# cnn + 评论数据 + 端到端
def model4(data_loader):
    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    semantic_feature_embed = SemanticFeatureEmbed(config)
    dnn_classifier = DNNClassifier(config)

    # loss and optimizer
    params = list(music_feature_extractor.parameters()) + list(intrinsic_feature_embed.parameters()) + list(semantic_feature_embed.parameters()) + list(dnn_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/4/{}".format(model_no)):
        os.makedirs("models/4/{}".format(model_no))

    # 训练模型
    losses = {}
    pos_neg_words = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for label, beta, mfcc, lyrics_vec, feature_words in tqdm(data_loader, ncols=80):
            # beta: (batch_size,)
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)
            # feature_words: (topk, batch_size) => 注意第一维不是batch_size

            batch_size = len(beta)
            topk = config.topk
            feature_words = feature_words[:topk]

            feature_words = np.array(feature_words).transpose() # 将feature_words两个维度调换
            # 将关键词转换为词向量表示
            feature_words_vec = torch.tensor(list(map(word_to_vec, feature_words.ravel()))).reshape(-1, 300)  # feature_words_vec: (batch_size*topk, 300)
            # 获取语义特征（关键词）的嵌入表示
            pos_embeds2 = semantic_feature_embed(feature_words_vec).reshape(batch_size, topk, -1) # pos_embeds2: (batch_size, topk, embedding_size)
            # 使用CNN网络提取音频特征
            refined_mfcc = music_feature_extractor(mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
            # 获得固有特征的嵌入表示
            embed1 = intrinsic_feature_embed(intrinsic_features) # embed1: (batch_size, embedding_size)
            # 输入分类器
            predict = dnn_classifier(embed1)

            # 计算损失
            loss1 = my_loss2(embed1, pos_embeds2, beta)
            loss2 = F.cross_entropy(predict, label)
            # loss = loss1 + loss2
            loss = loss1 + 10*loss2


            # zero_grad将上一次梯度更新的结果扔掉
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss1: {:.3f}, loss2: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss1.item(), loss2.item()))
                losses["{}-{}".format(epoch, step)] = (loss1.item(), loss2.item())

        # save models
        if epoch % config.save_epoch == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "models/4/{}/mf_extractor-e{}.pkl".format(model_no, epoch))
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/4/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(semantic_feature_embed.state_dict(), 
                        "models/4/{}/sf_embed-e{}.pkl".format(model_no, epoch))
            torch.save(dnn_classifier.state_dict(), 
                        "models/4/{}/dc_embed-e{}.pkl".format(model_no, epoch))
            with open("models/4/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)




# cnn + 评论数据 + 端到端 + 聚类
def model5(data_loader):
    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    # semantic_feature_embed = SemanticFeatureEmbed(config, in_size=)///
    dnn_classifier = DNNClassifier(config)

    # loss and optimizer
    params = list(music_feature_extractor.parameters()) + list(intrinsic_feature_embed.parameters()) + list(semantic_feature_embed.parameters()) + list(dnn_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/3/{}".format(model_no)):
        os.makedirs("models/3/{}".format(model_no))

    # 训练模型
    losses = {}
    pos_neg_words = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for label, beta, mfcc, lyrics_vec, feature_words in tqdm(data_loader, ncols=80):
            # beta: (batch_size,)
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)
            # feature_words: (topk, batch_size) => 注意第一维不是batch_size

            batch_size = len(beta)
            topk = config.topk
            feature_words = feature_words[:topk]

            feature_words = np.array(feature_words).transpose() # 将feature_words两个维度调换
            # 获得feature_words的聚类表示（向量）
            feature_cluster_vec = np.array([cluster_model(batch) for batch in feature_words]) # feature_cluster_vec: (batch_size, cluster_vec_size)
            # 获取语义特征的嵌入表示
            embed2 = semantic_feature_embed(feature_cluster_vec) # pos_embeds2: (batch_size, topk, embedding_size)
            # 使用CNN网络提取音频特征
            refined_mfcc = music_feature_extractor(mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整
            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)
            # 获得固有特征的嵌入表示
            embed1 = intrinsic_feature_embed(intrinsic_features) # embed1: (batch_size, embedding_size)
            # 输入分类器
            predict = dnn_classifier(embed1)

            # 计算损失
            loss = my_loss2(embed1, embed2, beta) + F.cross_entropy(predict, label)

            # zero_grad将上一次梯度更新的结果扔掉
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss.item()))
                losses["{}-{}".format(epoch, step)] = loss.item()

        # save models
        if epoch % config.save_epoch == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "models/5/{}/mf_extractor-e{}.pkl".format(model_no, epoch))
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/5/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(semantic_feature_embed.state_dict(), 
                        "models/5/{}/sf_embed-e{}.pkl".format(model_no, epoch))
            torch.save(dnn_classifier.state_dict(), 
                        "models/5/{}/dc_embed-e{}.pkl".format(model_no, epoch))
            with open("models/5/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)




def main():
    # data loader
    dataset, data_loader = get_loader(config)
    model = model2(data_loader)


if __name__ == '__main__':
    main()