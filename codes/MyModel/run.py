import os
import sys
import json
import pickle
import numpy as np 
from time import time
from tqdm import tqdm
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from data_loader import get_data_loader
from config import Config
from model import CustomCNN, DNNClassifier, MusicEmbed, TextCNNEmbed, VGG
from model_utils import get_acc, save_settings, embedding_loss_euclidean, get_contrastive_loss_kiros

import warnings 
warnings.filterwarnings("ignore")

class Runner:
    '''
    测试模型的父类
    '''
    def __init__(self, config, model_mode, model_index):
        self._build_model()
        self.config = config
        self.optimizer = torch.optim.SGD(self.params, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=config.FACTOR, patience=config.PATIENCE, verbose=True)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[50,70,90], gamma=0.1)
        self.model_save_dir = save_settings(model_mode, model_index, config) # 保存路径

    # 初始化模型结构和参数
    def _build_model(self):
        pass

    # 定义数据在模型中的流转，得到outputs
    def _model(self, batch_data):
        pass

    def _loss(self, batch_data):
        pass


    # 运行一个 epoch
    def run(self, data_loader, mode="train"):
        for model in self.models.values():
            model.train() if mode=="train" else model.eval()
        epoch_losses = []
        epoch_accs = []
        for batch_data in tqdm(data_loader):
            labels = batch_data["label"]
            loss, outputs = self._loss(batch_data)
            
            acc = get_acc(labels, outputs) # 计算本轮的精确度
            epoch_accs.append(acc)
            epoch_losses.append(loss.item())
            # print(loss)

            if mode == "train":
                self.optimizer.zero_grad() # 将上一步的梯度清零
                loss.backward() # 利用损失函数，反向传播计算梯度
                self.optimizer.step() # 利用梯度和学习率更新参数

        return epoch_losses, epoch_accs

    # Early stopping function for given validation loss
    def early_stop(self, loss):
        self.scheduler.step(loss)
        learning_rate = self.optimizer.param_groups[0]['lr']
        stop = learning_rate < self.config.STOPPING_RATE

        return stop

    # 保存loss和acc的变化图
    def save_loss_acc(self, data, step_size, notes):
        fig, axs = plt.subplots(2,2)
        for i, (k, v) in enumerate(data.items()):
            axs[i//2][i%2].plot(v)
            axs[i//2][i%2].set_title(k)
            if k in ("train_losses", "train_accs"):
                size = step_size[0]
            else:
                size = step_size[1]
            x = range(size//2, len(v), size)
            y = np.mean(np.array(v).reshape(-1, size), axis=1)
            axs[i//2][i%2].plot(x, y, color="orange")

        fig.suptitle(notes)
        fig.savefig(os.path.join(self.model_save_dir, "loss_acc_{}.png".format(notes)))
        plt.close()


    def save_models(self, epoch):
        for name, model in self.models.items():
            if name=="vggish": continue
            torch.save(model.state_dict(), 
                os.path.join(self.model_save_dir, "{}-e{}.pth".format(name, epoch)))



class MLA_FC(Runner):
    '''
    使用音频、歌词、艺人信息数据，直接使用全部的rawmusic数据（拼接在一起）。
    '''
    def __init__(self, config, model_index, music_feature_len):
        model_mode = "mla_fc" 
        self.config = config
        self.music_feature_len = music_feature_len
        self.dnn_in_init = music_feature_len + 372
        super().__init__(config, model_mode, model_index)


    def _build_model(self):
        self.models = {"dnn_classifier": DNNClassifier(self.config, key="fc_classifier")}
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

    def _loss(self, batch_data):
        music_features = batch_data["music_vec"].view(-1, self.music_feature_len) # 将频谱矩阵直接拉平
        X = torch.cat([music_features, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        outputs = self.models["dnn_classifier"](X)
        loss = F.cross_entropy(outputs, batch_data["label"])
        return loss, outputs


class MLA_CNN(Runner):
    '''
    使用音频、歌词、艺人信息数据，使用自定义CNN提取音频特征（来自melspectrogram）。
    '''
    def __init__(self, config, model_index):
        model_mode = "mla" 
        self.config = config
        super().__init__(config, model_mode, model_index)


    def _build_model(self):
        self.models = {
            "custom_cnn": CustomCNN(self.config),
            "dnn_classifier": DNNClassifier(self.config, DNN_IN_INIT=2420)
        }
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

    def _model(self, batch_data):
        music_features = self.models["custom_cnn"](batch_data["music_vec"])
        X = torch.cat([music_features, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        outputs = self.models["dnn_classifier"](X)
        return outputs



class MLAR(Runner):
    def __init__(self, config, model_index):
        model_mode = "mlar"
        self.config = config
        super().__init__(config, model_mode, model_index)

    def _build_model(self):
        self.models = {
            "music_embed": MusicEmbed(self.config),
            "textcnn_embed": TextCNNEmbed(self.config),
            "dnn_classifier": DNNClassifier(self.config, key="embed_classifier")
        }
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

    def _loss(self, batch_data):
        music_embed = self.models["music_embed"](batch_data["music_vec"])
        reviews_embed = self.models["textcnn_embed"](batch_data["reviews_vec"])
        X = torch.cat([music_embed, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        # X = torch.cat([music_embed, reviews_embed, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        outputs = self.models["dnn_classifier"](X)
        # loss = F.cross_entropy(outputs, batch_data["label"])
        loss = embedding_loss_euclidean(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # loss = get_contrastive_loss_kiros()(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # print(loss, outputs)
        return loss, outputs



class M2LAR(Runner):
    def __init__(self, config, model_index):
        model_mode = "m2lar"
        self.config = config
        super().__init__(config, model_mode, model_index)    


    def _build_model(self):
        self.models = {
            "music_embed": MusicEmbed(self.config),
            "textcnn_embed": TextCNNEmbed(self.config),
            "dnn_classifier": DNNClassifier(self.config, key="embed_classifier2")
        }
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

    def _loss(self, batch_data):
        music_embed = self.models["music_embed"](batch_data["music_vec"])
        reviews_embed = self.models["textcnn_embed"](batch_data["reviews_vec"])
        music_features = batch_data["music_vec"].view(-1, self.config.MUSICEMBED_IN_SIZE)
        X = torch.cat([music_features, music_embed, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        outputs = self.models["dnn_classifier"](X)
        # loss = F.cross_entropy(outputs, batch_data["label"])
        loss = embedding_loss_euclidean(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # loss = get_contrastive_loss_kiros()(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # print(loss, outputs)
        return loss, outputs



class VGG_MLAR(Runner):
    def __init__(self, config, model_index):
        model_mode = "vgg_mlar"
        self.config = config
        super().__init__(config, model_mode, model_index)

    def _build_model(self):
        self.models = {
            "vgg": VGG(self.config),
            "music_embed": MusicEmbed(self.config),
            "textcnn_embed": TextCNNEmbed(self.config),
            "dnn_classifier": DNNClassifier(self.config, key="embed_classifier")
        }
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

        # 使用原始VGGish模型参数初始化本地VGG参数
        pretrained_vgg = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=True)
        model_dict = self.models["vgg"].state_dict()
        pretrained_dict = {k: v for k, v in pretrained_vgg.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models["vgg"].load_state_dict(model_dict)


    def _loss(self, batch_data):
        vggish_examples = batch_data["music_vec"]
        s = vggish_examples.shape
        vggish_examples = vggish_examples.view(s[0]*s[1], s[2], s[3])
        vggish = self.models["vgg"](vggish_examples).view(s[0], -1)
        music_embed = self.models["music_embed"](vggish)
        reviews_embed = self.models["textcnn_embed"](batch_data["reviews_vec"])
        X = torch.cat([music_embed, batch_data["lyrics_vec"], batch_data["artist_vec"]], dim=1)
        outputs = self.models["dnn_classifier"](X)
        loss = embedding_loss_euclidean(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # loss = get_contrastive_loss_kiros()(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        # print(loss, outputs)
        return loss, outputs




class TextCNNOnly(Runner):
    def __init__(self, config, model_index):
        model_mode = "textcnn"
        self.config = config
        super().__init__(config, model_mode, model_index)

    def _build_model(self):
        self.models = {
            "classifier": TextCNNEmbed(self.config)
        }
        self.params = []
        for model in self.models.values():
            self.params += list(model.parameters())

    def _loss(self, batch_data):
        outputs = self.models["classifier"](batch_data["reviews_vec"])
        loss = F.cross_entropy(outputs, batch_data["label"])
        # loss = embedding_loss_euclidean(music_embed, reviews_embed) + F.cross_entropy(outputs, batch_data["label"])
        return loss, outputs




def main():
    config = Config()
    
    config.MUSIC_DATATYPE = "vggish"
    config.REVIEWS_VEC_KEY = "candidates_cls"
    # config.REVIEWS_VEC_KEY = "candidates"
    train_dataloader, valid_dataloader, test_dataloader = get_data_loader(config)
    # music_feature_len = 2560 # 2560 / 8620 / 17240
    # runner = MLA_FC(config, model_index, music_feature_len=music_feature_len)

    model_index = "0406"
    # notes = "textcnn only K=[4, 4] topk=5 candidates"
    notes = "mlar [4,4,4] embed=512 candidates_cls 3"
    # notes = "vgg_mlar [4, 4] candidates_cls 4"
    print("[NOTES]", notes)
    config.TEXTCNN_KERNEL_SIZES = [4]
    # config.TEXTCNN_NUM_CHANNELS = 512
    config.EMBED_SIZE = 512
    config.DNNCLASSIFIER_IN_SIZE = 628
    # config.DNNCLASSIFIER_IN_SIZE = 3188
    config.MUSICEMBED_IN_SIZE = 2560
    # config.WORD_EMBED_SIZE = 301
    runner = MLAR(config, model_index)
    # runner = TextCNNOnly(config, model_index)
    # runner = M2LAR(config, model_index)
    # runner = VGG_MLAR(config, model_index)

    records = {"train_losses":[], "valid_losses":[], "train_accs":[], "valid_accs":[]}
    for epoch in range(1, config.EPOCHS_NUM):
        epoch_train_losses, epoch_train_accs = runner.run(train_dataloader, "train")
        epoch_valid_losses, epoch_valid_accs = runner.run(valid_dataloader, "valid")
        # 这一部分好冗余，但是我要看一下曲线变化才行
        records["train_losses"].extend(epoch_train_losses)
        records["valid_losses"].extend(epoch_valid_losses)
        records["train_accs"].extend(epoch_train_accs)
        records["valid_accs"].extend(epoch_valid_accs)
        runner.save_loss_acc(records, step_size=(79, 10), notes=notes) # ⚠️

        # 打印/绘制训练信息
        print("[Epoch {}/{}] [Train Loss: {:.3f}] [Train Acc: {:.3f}] [Valid Loss: {:.3f}] [Valid Acc: {:.3f}]".format(
            epoch, config.EPOCHS_NUM, np.mean(epoch_train_losses), np.mean(epoch_train_accs), np.mean(epoch_valid_losses), np.mean(epoch_valid_accs)))

        # runner.print_params()
        # save models
        # if epoch % config.EPOCH_SAVE_STEP == 0:
            # runner.save_models(epoch)
            
        if runner.early_stop(np.mean(epoch_valid_losses)):
            # runner.save_models(epoch)
            break


if __name__ == '__main__':
    main()