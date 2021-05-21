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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from data_loader import get_data_loader
from config import Config
from model import DNNClassifier, MusicEmbed, TextCNNEmbed, VGG
from model_utils import get_acc, save_settings, embedding_loss_euclidean, embedding_loss_dot_product, \
                        embedding_loss_contrastive, musicvgg_load_pretrained_params, embedding_loss_contrastive_ml

import warnings 
warnings.filterwarnings("ignore")

class Runner:
    '''
    测试模型的父类
    '''
    def __init__(self, config):
        self._build_model()
        self.config = config
        self.optimizer = torch.optim.SGD(self.params, lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        if config.SCHEDULER=="cyclic":
            if self.model_mode=="vgg_mlar":
                base_lr=[3e-6, 3e-4, 3e-4, 3e-4]
                max_lr=[1.5e-5, 1.5e-3, 1.5e-3, 1.5e-3]
            elif self.model_mode=="vgg_mla":
                base_lr=[3e-6, 3e-4]
                max_lr=[1.5e-5, 1.5e-3]                
            self.scheduler = CyclicLR(
                self.optimizer, 
                base_lr=base_lr, max_lr=max_lr, 
                step_size_up=200, # 推荐2-10倍的epochs（此处的step表示一个iter）
                # mode="exp_range", # 指数减小学习率
                # gamma=0.9999, # 和exp_range配合使用，否则默认为1，不会有指数衰减的效果
            )
        elif config.SCHEDULER=="reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=config.FACTOR, 
                                               patience=config.PATIENCE, verbose=True)        
        self.model_save_dir = save_settings(self.model_mode, self.model_index, config) # 保存路径

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
            labels = batch_data["label"].to(self.device)
            loss, outputs = self._loss(batch_data)
            
            acc = get_acc(labels, outputs) # 计算本轮的精确度
            epoch_accs.append(acc)
            epoch_losses.append(loss.item())
            # print(loss)

            if mode == "train":
                self.optimizer.zero_grad() # 将上一步的梯度清零
                loss.backward() # 利用损失函数，反向传播计算梯度
                self.optimizer.step() # 利用梯度和学习率更新参数
                if self.config.SCHEDULER=="cyclic":
                    self.scheduler.step() # cyclic调度器每个step执行一次

        return epoch_losses, epoch_accs



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


    
class TextCNNOnly(Runner):
    '''
    + 仅使用评论文本数据进行分类。
    + 模型结构：
        - textcnn(comments) => dnn_classifier
    '''
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

    
    

class VGG_MLA(Runner):
    '''
    联合表征，音频特征使用 VGGish 微调。
    '''
    def __init__(self, config, model_index, device):
        self.model_mode = "vgg_mla"
        self.model_index = model_index
        self.config = config
        self.device = device
        super().__init__(config)

    def _build_model(self):
        # 模型模块
        self.models = {
            "vgg": VGG().to(self.device),
            "dnn_classifier": DNNClassifier(self.config, key="vgg_mla").to(self.device)
        }
        # 模型参数
        self.params = []
        for k, v in self.models.items():
            if k=="vgg":
                if not self.config.VGG_STATIC: 
                    self.params += [{"params": v.parameters(), "lr":self.config.VGG_LR}]
                continue
            self.params += [{"params": v.parameters()}]

        # 使用原始VGGish模型参数初始化本地VGG参数
        pretrained_vgg = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=True)
        model_dict = self.models["vgg"].state_dict()
        pretrained_dict = {k: v for k, v in pretrained_vgg.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models["vgg"].load_state_dict(model_dict)

    def _loss(self, batch_data):
        music_vec, reviews_vec, lyrics_vec, artist_vec, labels =\
         batch_data["music_vec"].to(self.device), batch_data["reviews_vec"].to(self.device), \
         batch_data["lyrics_vec"].to(self.device), batch_data["artist_vec"].to(self.device),  batch_data["label"].to(self.device)

        s = music_vec.shape #（batch_size, 20, 96, 64）
        music_vec = music_vec.contiguous().view(s[0]*s[1], s[2], s[3]) # (batch_size*20, 96, 64)
        vggish = self.models["vgg"](music_vec) # (batch_size*20, 128)
        vggish = vggish.contiguous().view(s[0], -1) # (batch_size, 20*128)

        X = torch.cat([vggish, lyrics_vec, artist_vec], dim=1)
        outputs = self.models["dnn_classifier"](X)
        loss = F.cross_entropy(outputs, labels)

        return loss, outputs    
    
    


class VGG_MLAR(Runner):
    '''
    双表征。音频特征经过 VGGish 微调，评论特征经过的 TextCNN 模型，分别嵌入协同空间。
    将得到的协同音频特征与歌词特征、艺人特征进行联合表示，并送入分类器。
    '''
    def __init__(self, config, model_index, device):
        self.model_mode = "vgg_mlar"
        self.model_index = model_index
        self.config = config
        self.device = device
        super().__init__(config)

    def _build_model(self):
        # 模型模块
        self.models = {
            "vgg": VGG().to(self.device),
            "music_embed": MusicEmbed(self.config).to(self.device), # 嵌入模型
            "textcnn_embed": TextCNNEmbed(self.config).to(self.device), # textcnn+嵌入模型
            "dnn_classifier": DNNClassifier(self.config, key="vgg_mlar").to(self.device)
        }
        
        # 模型参数
        self.params = []
        for k, v in self.models.items():
            if k=="vgg":
                if not self.config.VGG_STATIC: 
                    self.params += [{"params": v.parameters(), "lr":self.config.VGG_LR}]
                continue
            self.params += [{"params": v.parameters()}]

        # 使用原始VGGish模型参数初始化本地VGG参数
        pretrained_vgg = torch.hub.load("harritaylor/torchvggish", "vggish", pretrained=True)
        model_dict = self.models["vgg"].state_dict()
        pretrained_dict = {k: v for k, v in pretrained_vgg.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models["vgg"].load_state_dict(model_dict)


    def _loss(self, batch_data):
        music_vec, reviews_vec, lyrics_vec, artist_vec, labels =\
         batch_data["music_vec"].to(self.device), batch_data["reviews_vec"].to(self.device), \
         batch_data["lyrics_vec"].to(self.device), batch_data["artist_vec"].to(self.device),  batch_data["label"].to(self.device)

        s = music_vec.shape #（batch_size, 20, 96, 64）
        music_vec = music_vec.contiguous().view(s[0]*s[1], s[2], s[3]) # (batch_size*20, 96, 64)
        vggish = self.models["vgg"](music_vec) # (batch_size*20, 128)
        vggish = vggish.contiguous().view(s[0], -1) # (batch_size, 20*128)

        music_embed = self.models["music_embed"](vggish)
        reviews_embed = self.models["textcnn_embed"](reviews_vec)

        X = torch.cat([music_embed, lyrics_vec, artist_vec], dim=1)
        outputs = self.models["dnn_classifier"](X)
        loss = embedding_loss_euclidean(music_embed, reviews_embed) + F.cross_entropy(outputs, labels)
        # loss = embedding_loss_dot_product(music_embed, reviews_embed) + F.cross_entropy(outputs, labels)    
        # loss = embedding_loss_contrastive_ml(mode="lp")(
        #           torch.cat((music_embed, reviews_embed)), torch.Tensor(list(range(s[0]))+list(range(s[0])))) \
        #        + F.cross_entropy(outputs, labels)
        
        return loss, outputs





def main():
    config = Config()
    
    # 加载数据集
    config.MUSIC_DATATYPE = "vggish_examples"
    config.REVIEWS_VEC_KEY = "candidates_cls"
    train_dataloader, test_dataloader = get_data_loader(config)
    
    # 使用设备（使用用显卡加速）
    os.environ['CUDA_VISIBLE_DEVICES']='3' # 似乎会和在cuda:0上跑tensorflow的同学冲突
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 调整模型配置信息
    config.SCHEDULER = "reduce_on_plateau"
    # config.SCHEDULER = "cyclic"
    config.TEXTCNN_KERNEL_SIZES = [4, 4]
    # config.TEXTCNN_NUM_CHANNELS = 512
    config.EMBED_SIZE = 256
    config.VGG_STATIC = False # 是否保持VGG模型的参数不变
    config.EPOCHS_NUM = 300
    config.DNNCLASSIFIER_DROPOUT_RATE = 0.2
    config.TEXTCNN_DROPOUT_RATE = 0.2
    config.MUSICEMBED_DROPOUT_RATE = 0.2


    # 模型
    model_index = "0513"
    # notes = "textcnn only K=[4, 4] topk=5 candidates"
    # notes = "mlar [4,4,4] embed=512 candidates_cls 3"
    # notes = "vgg_mlar dynamic [4, 4] candidates_cls 4"
    # notes = "vgg_mlar dynamic [4,4] candidates_cls reduce_on_plateau neg2 LP" 

    
    runner = VGG_MLAR(config, model_index, device)
    # runner = VGG_MLA(config, model_index, device)
    logfile = "log1.txt"
    
    notes = "{} {} {} {} {}".format(
        runner.model_mode, # 模型名称
        config.SCHEDULER, # 学习率调度器
        "LP", # 度量方式
        "neg2", # 数据集
        "dp2_all" # 其他信息
    )
    print("[NOTES]", notes)
    with open(logfile, "a") as f:
        f.write('\n'+notes+'\n')

    # 训练
    records = {"train_losses":[], "test_losses":[], "train_accs":[], "test_accs":[]}
    for epoch in range(1, config.EPOCHS_NUM):
        # 得到训练集和测试集的 loss, acc
        epoch_train_losses, epoch_train_accs = runner.run(train_dataloader, "train")
        epoch_test_losses, epoch_test_accs = runner.run(test_dataloader, "test")
        
        # 将 loss, acc 信息保存下来，并进行绘制
        train_iters, test_iters = len(train_dataloader), len(test_dataloader) # 训练和测试分别需要的iteration数
        records["train_losses"].extend(epoch_train_losses)
        records["test_losses"].extend(epoch_test_losses)
        records["train_accs"].extend(epoch_train_accs)
        records["test_accs"].extend(epoch_test_accs)
        runner.save_loss_acc(records, step_size=(train_iters, test_iters), notes=notes)

        # 打印/绘制训练信息
        info_format = "[Epoch {}/{}] [Train Loss: {:.3f}] [Train Acc: {:.3f}] [Test Loss: {:.3f}] [Test Acc: {:.3f}] LR: {:.7f}, {:.7f}" # 记录学习率变化信息
        info_format = "[Epoch {}/{}] [Train Loss: {:.3f}] [Train Acc: {:.3f}] [Test Loss: {:.3f}] [Test Acc: {:.3f}]"
        info = info_format.format(
                epoch, config.EPOCHS_NUM, np.mean(epoch_train_losses), np.mean(epoch_train_accs), 
                np.mean(epoch_test_losses), np.mean(epoch_test_accs),) 
                # runner.scheduler.get_lr()[0], runner.scheduler.get_lr()[1])
        print(info)
        with open(logfile,"a") as f:
            f.write(info+'\n')

        if config.SCHEDULER == "reduce_on_plateau":
            runner.scheduler.step(np.mean(epoch_test_losses)) # reduce_on_plateau每个epoch使用一次


if __name__ == '__main__':
    main()
