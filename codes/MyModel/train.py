import os
import sys
import json
import pickle
import numpy as np 
from time import time
from tqdm import tqdm
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_data_loader
from config import Config
from model import MusicFeatureExtractor, DNNClassifier
# from model_utils import get_pairwise_embed, my_loss1, my_loss2, my_loss3

config = Config()


def acc(y, y_pred):
    # y_pred = F.softmax(y_pred)
    corrects = (torch.max(y_pred, 1)[1].view(-1).data == y.data).sum()
    # print("Predicted correctly: {}/{} = {:.2f}%".format(
    #     corrects, len(y), float(corrects)/len(y)*100))
    return float(corrects)/len(y)*100


def model1(train_data_loader, test_data_loader):
    '''
    使用 music, lyrics, artist 特征
    '''
    # models
    music_feature_extractor = MusicFeatureExtractor(config)
    dnn_classifier = DNNClassifier(config)    
    # 其他设定
    model_mode = 1
    model_code = int(time())
    params = list(music_feature_extractor.parameters()) + list(dnn_classifier.parameters())
    optimizer = torch.optim.SGD(params, lr=config.LR) # 优化器使用Adam
    # 保存路径
    model_save_path = "models/{}/{}".format(model_mode, model_code)
    config_save_path = "models/{}/{}/config.json".format(model_mode, model_code)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    with open(config_save_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # 训练模型
    BATCH_SIZE = config.TRAIN_BATCH_SIZE
    total_step = config.TRAIN_DATASET_SIZE // BATCH_SIZE + 1
    for epoch in range(1, config.NUM_EPOCHS+1):
        # 遍历batch
        step = 0
        for batch_data in tqdm(train_data_loader, ncols=50):
            labels = batch_data["label"]
            music_feature = music_feature_extractor(batch_data["mfcc_vec"])
            lyrics_feature = batch_data["lyrics_vec"]
            artist_feature = batch_data["artist_vec"]

            features = torch.cat([music_feature, lyrics_feature, artist_feature], dim=1)
            outputs = dnn_classifier(features)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
            # 一个batch结束，输出当前loss
            print("epoch[{}/{}], step[{}/{}], loss: {:.3f}, acc: {:.2f}%".format(
                epoch, config.NUM_EPOCHS, step, total_step, loss.item(), acc(labels, outputs)))
            # losses["{}-{}".format(epoch, step)] = loss.item()       

        # 一个epoch结束，输出当前参数在整体样本上的测试结果
        # pass
        
        # save models
        if epoch % config.EPOCH_SAVE_STEP == 0:
            torch.save(music_feature_extractor.state_dict(), 
                        "{}/mf_extractor-e{}.pkl".format(model_save_path, epoch))
            torch.save(dnn_classifier.state_dict(), 
                        "{}/dc_embed-e{}.pkl".format(model_save_path, epoch))



def main():
    # data loader
    train_data_loader = get_data_loader(config, mode="train", batch_size=config.TRAIN_BATCH_SIZE) # 不同mode决定数据集的不同
    test_data_loader = get_data_loader(config, mode="test", batch_size=config.TEST_BATCH_SIZE)
    model = model1(train_data_loader, test_data_loader)


if __name__ == '__main__':
    main()