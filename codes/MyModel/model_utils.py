import os
import json
import numpy as np 
import pickle
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance


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
    计算嵌入损失，最小化欧式距离
    '''
    loss = torch.mean(F.pairwise_distance(embed1, embed2))
    return loss


def embedding_loss_dot_product(embed1, embed2):
    '''
    计算嵌入损失，最大化点积相似度
    '''
    # 需要先标准化，防止loss无穷大
    embed1_norm = F.normalize(embed1, p=2, dim=1)
    embed2_norm = F.normalize(embed2, p=2, dim=1)
    # torch.sum(embed1*embed2, dim=1) 计算各匹配特征的点积
    # 1-x 将最大化点击相似度转换为最小化损失函数问题
    loss = torch.mean(1-torch.sum(embed1_norm*embed2_norm, dim=1))   
    return loss



def embedding_loss_contrastive(device, gamma=0, symmetric=False):
    '''
    contrasive loss（对比损失），使用点积相似度。
    Args:
        gamma: 间隔（要求匹配的相似度至少比不匹配的大多少）
        symmetric: 无作用
    '''

    def loss(embed1, embed2):
        '''
        contrasive loss 的具体计算过程。
        Args:
            embed1, embed2: 待优化距离的向量组，分别对应音频和文本，大小为(N, embed_size)
        '''

        # 需要先标准化，防止loss无穷大
        embed1_norm = F.normalize(embed1, p=2, dim=1)
        embed2_norm = F.normalize(embed2, p=2, dim=1)        

        # 计算音频嵌入和文本嵌入的相似度矩阵
        scores = torch.matmul(embed1_norm, embed2_norm.T)
        diagonal = scores.diagonal() # 取出对角线上的值（配对）
        zero_mtx = torch.zeros(scores.shape).to(device)
        
        # 将对角线上的值和其所在的列进行比较（列对应的是和某个文本最匹配的音频）
        cost_s = torch.max(zero_mtx, gamma - diagonal + scores)
        # 将对角线上的值和其所在的行进行比较（行对应的是和某个音频最匹配的文本）
        cost_im = torch.max(zero_mtx, gamma - diagonal.reshape((-1, 1)) + scores)

        # 清除对角线上的gamma值
        cost_s.fill_diagonal_(0)
        cost_im.fill_diagonal_(0)

        return cost_s.sum() + cost_im.sum()

    return loss



def embedding_loss_contrastive_ml(mode="lp"):
    '''
    使用pytorch-metric-learning中的contrasive loss，距离默认为欧式距离（LpDistance）
    '''
    if mode=="lp":
        loss = ContrastiveLoss(
            pos_margin=0.25, neg_margin=1.5, distance=LpDistance(power=2)
        )
    elif mode=="cos":
        loss = ContrastiveLoss(
            pos_margin=1.5, neg_margin=0.6, distance=CosineSimilarity()
        )        
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


def model_size_in_gpu(model, input, type_size=4):
    # 模型显存占用监测函数
    # model：输入的模型
    # input：实际中需要输入的Tensor变量
    # type_size 默认为 4 默认类型为 float32 

#     print(model.parameters)
    para = sum([np.prod(list(p.size())) for p in model.parameters()]) # 总参数量
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
#     input_.requires_grad_(requires_grad=False)
    input_.detach()

    mods = list(model.modules())
#     print(len(mods))
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.Sequential): continue
        if isinstance(m, nn.ReLU) and m.inplace: continue
        out = m(input_)
        out_sizes.append(np.array(out.size())) # 保存每一层的中间变量大小
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


