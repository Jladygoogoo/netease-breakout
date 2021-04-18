import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


CNN_framework = {
    "mel": [64, "M-2", 128, "M-2", 256, "M-4"],
    "vggish": [64, "M-2", 128, "M-2", 256]
}

dnn_hiddens = {
    "fc_classifier": [1024, 512, 128],
    "embed_classifier": [256, 128, 64],
    "embed_classifier2": [1024, 512, 64],
    "music_embed": [1024, 512, 256]
}

def make_layers_customcnn(k):
    layers = []
    in_channels = 1
    for v in CNN_framework[k]:
        if type(v) == str:
            s = int(v.split('-')[-1])
            layers += [nn.MaxPool2d(kernel_size=s, stride=s)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_vgg():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_vgg2():
    # musicnn: pre-trained convolutional neural networks for music audio tagging
    layers = []
    maxpool_set = [(4,1,2),(2,2,2),(2,2,2),(2,2,2),(4,4,4)]
    in_channels = 1

    for s in maxpool_set:
        conv2d = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 128
        layers += [nn.BatchNorm2d(num_features=in_channels)]
        layers += [nn.MaxPool2d(kernel_size=(s[0],s[1]), stride=(s[2],s[2]))]

    return nn.Sequential(*layers)


class CustomCNN(nn.Module):
    '''
    一个CNN模型
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        DYNAMICS_IN_FEATURES = 53248 # 计算麻烦，直接拿一条数据输出看一下
        self.custom_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4),  
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=DYNAMICS_IN_FEATURES, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x.shape: (BATCH_SIZE, N_FEATURE, FRAMES_NUM)
        '''
        x = x.unsqueeze(1) # x.shape: (BATCH_SIZE, 1, N_FEATURE, FRAMES_NUM)
        x = self.custom_cnn(x)
        x = x.view(x.size(0), -1) # x.shape: (BATCH_SZIE, -1)
        # print(x.shape)
        x = self.fc(x) # x.shape: (BATCH_SIZE, CNN_OUT_LENGTH)

        return x

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)



class MusicVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = make_layers_vgg2()
        self.fc = nn.Linear(in_features=2*128, out_features=50) # 原模型输出音乐种类数为50

    def forward(self, x):
        x = x.unsqueeze(1) # 增加channel纬度 (batch_size, 1, n_frames, n_mels)
        x = self.features(x) # 经过卷积网络
        x = x.view(x.shape[0], -1) # (batch_size, -1)
        return x




class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = make_layers_vgg()
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)




class DNNClassifier(nn.Module):
    '''
    使用简单前馈网络作为分类器
    '''
    def __init__(self, config, key):
        super().__init__()
        layers = []
        hiddens = [config.DNNCLASSIFIER_IN_SIZE] + dnn_hiddens[key]
        for i in range(len(hiddens)-1):
            layers.extend([nn.Linear(hiddens[i], hiddens[i+1]), 
                           nn.ReLU(), 
                           nn.Dropout(config.DNNCLASSIFIER_DROPOUT_RATE)])
        layers.append(nn.Linear(hiddens[-1], config.NUM_CLASS))
        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        outputs = self.classifier(x)
        return outputs


class MusicEmbed(nn.Module):
    '''
    直接使用全连接层对音频特征进行嵌入
    '''
    def __init__(self, config):
        super().__init__()
        hiddens = [config.MUSICEMBED_IN_SIZE] + dnn_hiddens["music_embed"]
        layers = []
        for i in range(len(hiddens)-2):
            layers.extend([
                nn.Linear(hiddens[i], hiddens[i+1]), nn.ReLU(), nn.Dropout(config.MUSICEMBED_DROPOUT_RATE)])
        layers.append(nn.Linear(hiddens[-2], hiddens[-1]))
        self.embed = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) # x.shape: (BATCH_SZIE, -1)
        output = self.embed(x)
        return output



class TextCNNEmbed(nn.Module):
    '''
    使用textCNN对评论文本进行嵌入
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList(
            [nn.Conv1d(config.WORD_EMBED_SIZE, config.TEXTCNN_NUM_CHANNELS, w) for w in config.TEXTCNN_KERNEL_SIZES])
        self.fc = nn.Linear(config.TEXTCNN_NUM_CHANNELS*len(config.TEXTCNN_KERNEL_SIZES), config.EMBED_SIZE)
        print("model", config.TEXTCNN_NUM_CHANNELS*len(config.TEXTCNN_KERNEL_SIZES))
        # self.fc = nn.Sequential(
        #     nn.Linear(config.TEXTCNN_NUM_CHANNELS*len(config.TEXTCNN_KERNEL_SIZES), config.EMBED_SIZE),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.01),
        #     nn.Linear(config.EMBED_SIZE, 128),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.01),
        #     nn.Linear(128, 2)
        # )

    def forward(self, x):
        # x = x.unsqueeze(1) # 增加维度 num_channel=1
        x = x.permute(0,2,1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(xi, xi.size(2)).squeeze(2) for xi in x]
        x = torch.cat(x, 1)
        x = F.dropout(x, p=self.config.TEXTCNN_DROPOUT_RATE)
        output = self.fc(x)
        return output





