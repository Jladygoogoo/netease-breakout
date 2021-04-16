import os
import json
import numpy as np 
import pickle
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MusicVGG


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


def get_acc(y, y_pred):
    # y_pred = F.softmax(y_pred)
    corrects = (torch.max(y_pred, 1)[1].view(-1).data == y.data).sum()
    # print("Predicted correctly: {}/{} = {:.2f}%".format(
    #     corrects, len(y), float(corrects)/len(y)*100))
    return float(corrects)/len(y)


def save_settings(model_mode, model_index, config):
    model_save_dir = "models/{}/{}".format(model_mode, model_index)
    config_save_path = "models/{}/{}/config.json".format(model_mode, model_index)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    with open(config_save_path, "w") as f:
        json.dump(config.__dict__, f, indent=4)

    return model_save_dir



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


def embedding_loss_euclidean(embed1, embed2):
    '''
    使用欧式距离
    '''
    loss = torch.mean(F.pairwise_distance(embed1, embed2))
    return loss



def get_contrastive_loss_kiros(gamma=0, symmetric=False):
    """ Compile contrastive loss (Kiros et al. 2014) """

    def loss(lv1, lv2):
        """ Contrastive cosine distance optimization target """

        # compute image-sentence score matrix
        scores = torch.matmul(lv1, lv2.T)
        diagonal = scores.diagonal()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(torch.zeros(scores.shape), gamma - diagonal + scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(torch.zeros(scores.shape), gamma - diagonal.reshape((-1, 1)) + scores)

        # clear diagonals
        cost_s.fill_diagonal_(0)
        cost_im.fill_diagonal_(0)

        return cost_s.sum() + cost_im.sum()

    return loss


def get_contrastive_cos_loss(weight, gamma, symmetric=False):
    """ Compile contrastive loss (Kiros et al. 2014) """

    def loss(lv1, lv2):
        """ Contrastive cosine distance optimization target """

        n = lv1.shape[0]

        # direction 1
        D = lv1.dot(lv2.T)
        d = D.diagonal().reshape((-1, 1))

        M = T.identity_like(D)
        O = D[(M <= 0).nonzero()].reshape((n, n - 1))

        L = gamma - d
        L = T.repeat(L, n - 1, 1)
        L += O
        L = T.clip(L, 0, 1000)

        loss = L.mean()

        # direction 2
        if symmetric:
            D = lv2.dot(lv1.T)
            d = D.diagonal().reshape((-1, 1))

            M = T.identity_like(D)
            O = D[(M <= 0).nonzero()].reshape((n, n - 1))

            L = gamma - d
            L = T.repeat(L, n - 1, 1)
            L += O
            L = T.clip(L, 0, 1000)

            loss += L.mean()

        return weight * loss

    return loss



def get_mel_3seconds_groups(audio_file, config, offset, duration):
    '''For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

    INPUT
    
    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'

    - n_frames: length (in frames) of the input spectrogram patches.
    Data format: integer.
    Example: 187
        
    - overlap: ammount of overlap (in frames) of the input spectrogram patches.
    Note: Set it considering n_frames.
    Data format: integer.
    Example: 10
    
    OUTPUT
    
    - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
    Data format: 3D np.array (batch, time, frequency)
    
    - audio_rep: raw audio representation (spectrogram).
    Data format: 2D np.array (time, frequency)
    '''

    # 3s对应的帧数
    n_frames = librosa.time_to_frames(3, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    overlap = n_frames # 不重叠

    audio, sr = librosa.load(audio_file, sr=config.SR, offset=offset, duration=duration)
    audio_rep = librosa.feature.melspectrogram(y=audio, 
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.NUM_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch



def musicvgg_load_pretrained_params(params_filepath, musicvgg):
    # 将原tensorflow模型MTT_VGG上的参数重载入本MusicVGG模型中
    with open(params_filepath, "rb") as f:
        tf_params = pickle.load(f)

    for k in musicvgg.state_dict():
        if "fc" in k:
            if "weight" in k:
                musicvgg.state_dict()[k] = torch.Tensor(tf_params["dense/kernel:0"])
            else:
                musicvgg.state_dict()[k] = torch.Tensor(tf_params["dense/bias:0"])
            continue

        layer_index = int(k.split('.')[1])
        if layer_index%4 != 0: # batchNorm
            continue 
        tf_cnn_index = layer_index // 4 + 1
        if "weight" in k:
            musicvgg.state_dict()[k] = torch.Tensor(tf_params["{}CNN/kernel:0".format(tf_cnn_index)])
        else:
            musicvgg.state_dict()[k] = torch.Tensor(tf_params["{}CNN/bias:0".format(tf_cnn_index)])



def musicnn_load_pretrained_params(params_filepath, musicnn):
    # 将原tensorflow模型MTT_CNN上的参数重载入本MusiCNN模型中
    with open(params_filepath, "rb") as f:
        tf_params = pickle.load(f) 


