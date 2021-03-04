import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F



class MusicFeatureExtractor(nn.Module):
    '''
    一个CNN模型
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.convs = []
        DYNAMICS_IN_FEATURES = 33280 # 计算麻烦，直接拿一条数据输出看一下
        for i in range(config.CNN_NUM_CONVS):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels=config.CNN_IN_C[i], out_channels=config.CNN_OUT_C[i], 
                    kernel_size=config.CNN_CONV_KERNEL_SIZE[i], stride=1, padding=1),
                nn.BatchNorm2d(num_features=config.CNN_OUT_C[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, config.CNN_MAX_KERNEL_SIZE[i]), stride=(1, config.CNN_MAX_KERNEL_SIZE[i]))
            ))

        self.fc = nn.Linear(in_features=DYNAMICS_IN_FEATURES, out_features=config.CNN_OUT_LENGTH)

    def forward(self, x):
        '''
        x.shape: (BATCH_SIZE, 20, FRAMES_NUM)
        '''
        x = x.unsqueeze(1) # x.shape: (BATCH_SIZE, 1, 20, FRAMES_NUM)
        for i in range(self.config.CNN_NUM_CONVS):
            x = self.convs[i](x)
        x = x.view(x.size(0), -1) # x.shape: (BATCH_SZIE, -1)
        x = F.dropout(x) # p=0.5(default) 
        x = self.fc(x) # x.shape: (BATCH_SIZE, CNN_OUT_LENGTH)

        return x




class DNNClassifier(nn.Module):
    '''
    使用简单前馈网络作为分类器
    '''
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for i in range(config.DNN_NUM_HIDDENS):
            self.layers.append(nn.Sequential(
                nn.Linear(config.DNN_IN[i], config.DNN_HIDDEN[i]),
                nn.ReLU(),
                nn.Dropout(),
            )) 
        self.layers.append(nn.Linear(config.DNN_HIDDEN[-1], config.NUM_CLASS)) # 输出分类结果

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x



