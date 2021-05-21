import os
import numpy as np
from config import Config
from data_loader import split_tracks_from_files
from model_utils import model_size_in_gpu
from model import VGG


def calc_vgg_size_in_gpu():
    # 计算 VGG 模型需占用的显存（粗略）
    config = Config()
    config.MUSIC_DATATYPE = "vggish_examples"
    config.REVIEWS_VEC_KEY = "candidates_cls"
    train_dataloader, valid_dataloader, test_dataloader = get_data_loader(config)
    for batch_data in train_dataloader:
        input_ = batch_data["music_vec"] # (batch_size, 20, 96, 64)
        s = input_.shape
        input_ = input_.view(s[0]*s[1], s[2], s[3])
        input_ = input_.unsqueeze(1)
        break
    vgg = VGG()
    model_size_in_gpu(vgg, input_, type_size=4)


if __name__ == '__main__':
    config = Config()
#     config.MUSIC_DATATYPE="vggish"
#     config.REVIEWS_VEC_KEY = "candidates"
#     train_dataloader, valid_dataloader, test_dataloader = get_data_loader(config)

    train_tracks, valid_tracks, test_tracks = split_tracks_from_files(
        2600, 300, 100, config.POS_TRACKS_FILEPATH, config.NEG_TRACKS_FILEPATH)
    print(train_tracks[:20])
    print(valid_tracks[:20])