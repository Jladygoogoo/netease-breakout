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
from model import MusicFeatureExtractor, IntrinsicFeatureEmbed, SemanticFeatureEmbed
from model_utils import get_pairwise_embed, my_loss1, my_loss2, my_loss3

# device configuratuon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configurations
config = Config()
w2v_model = Word2Vec.load(config.w2v_path)
def word_to_vec(word): # 得到词语的向量表示
    return w2v_model.wv[word] # 默认w2v_model.wv.__contains__(word)=True


def main():
    # data loader
    dataset, data_loader = get_loader(config.json_path, config.w2v_path, config.d2v_path, 
                        config.batch_size)

    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    intrinsic_feature_embed = IntrinsicFeatureEmbed(config)
    semantic_feature_embed = SemanticFeatureEmbed(config)

    # loss and optimizer
    params = list(music_feature_extractor.parameters()) + list(intrinsic_feature_embed.parameters()) + list(semantic_feature_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # 准备模型保存路径
    model_no = sys.argv[1]
    if not os.path.exists("models/{}".format(model_no)):
        os.makedirs("models/{}".format(model_no))

    # 训练模型
    losses = {}
    pos_neg_words = {}
    for epoch in range(1, config.num_epochs+1):
        step = 0
        total_step = len(data_loader) # 一个epoch中的batch数目
        for beta, mfcc, lyrics_vec, feature_words in tqdm(data_loader, ncols=80):
            # beta: (batch_size,)
            # mfcc: (batch_size, 20, 1292)
            # lyrics_vec: (batch_size, 300)
            # feature_words: (topk, batch_size) => 注意第一维不是batch_size

            # 没有将 word_2_vec 封装在 data_loader 里面是迫不得已的
            # pairwise 训练中，负样本的选取需要使用非向量的feature_words（向量unhashable）

            batch_size = len(beta)
            topk = config.topk
            feature_words = feature_words[:topk]

            feature_words = np.array(feature_words).transpose() # 将feature_words两个维度调换
            # 将关键词转换为词向量表示
            feature_words_vec = torch.tensor(list(map(word_to_vec, feature_words.ravel()))) # feature_words_vec: (batch_size*topk, 300)
            # 获取语义特征（关键词）的嵌入表示
            pos_embeds2 = semantic_feature_embed(feature_words_vec).reshape(batch_size, topk, -1) # pos_embeds2: (batch_size, topk, embedding_size)

            # 使用CNN网络提取音频特征
            refined_mfcc = music_feature_extractor(mfcc) # refined_mfcc: (batch_size, 1000) => 1024是fc的输出，可调整

            # 将音频特征和歌词特征拼接在一起
            intrinsic_features = torch.cat((refined_mfcc, lyrics_vec), axis=1) # intrinsic_features: (batch_size, 1324)

            # 获得固有特征的嵌入表示
            embed1 = intrinsic_feature_embed(intrinsic_features) # embed1: (batch_size, embedding_size)

            # 获取一个batch中所有样本点各自的负采样关键词的嵌入表示
            # semantic_feature_embed是当前的嵌入方式
            neg_embeds2, pos_neg_words_l = get_pairwise_embed(feature_words, embed1, semantic_feature_embed, word_to_vec)
            pos_neg_words["{}-{}".format(epoch, step)] = pos_neg_words_l

            # 计算损失
            # loss = my_loss1(embed1, pos_embeds2)
            loss = my_loss2(embed1, pos_embeds2, beta)
            # loss = my_loss3(embed1, pos_embeds2, neg_embeds2, beta)

            # zero_grad将上一次梯度更新的结果扔掉
            music_feature_extractor.zero_grad()
            intrinsic_feature_embed.zero_grad()
            semantic_feature_embed.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # log
            if step % config.log_step == 0:
                print("epoch [{}/{}], step [{}/{}], loss: {:.3f}".format(
                    epoch, config.num_epochs, step, total_step, loss.item()))
                losses["{}-{}".format(epoch, step)] = loss.item()

        print(losses)
        with open("models/pos_neg_words-{}.pkl".format(epoch), "wb") as f:
            pickle.dump(pos_neg_words, f)

        # save models
        if epoch % config.save_epoch == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "models/{}/mf_extractor-e{}.pkl".format(model_no, epoch))
            torch.save(intrinsic_feature_embed.state_dict(), 
                        "models/{}/if_embed-e{}.pkl".format(model_no, epoch))
            torch.save(semantic_feature_embed.state_dict(), 
                        "models/{}/sf_embed-e{}.pkl".format(model_no, epoch))
            with open("models/{}/losses.pkl".format(model_no), "wb") as f:
                pickle.dump(losses, f)



if __name__ == '__main__':
    main()