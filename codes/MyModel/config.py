class Config:
    def __init__(self):
        # 数据
        self.USE_ARTIST = True
        self.USE_REVIEWS = False
        self.LYRICS_VEC_LENGTH = 300
        self.ARTIST_VEC_LENGTH = 72 if self.USE_ARTIST else 0
        self.DATASET_OFFSET = 0
        self.TRAIN_DATASET_SIZE = 4000
        self.TEST_DATASET_SIZE = 200
        self.TRAIN_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 100
        self.NUM_CLASS = 2

        # 模型设定
        self.LR = 0.001
        self.NUM_EPOCHS = 10
        self.EPOCH_SAVE_STEP = 5
        ## CNN
        self.CNN_OUT_LENGTH = 1024
        self.CNN_NUM_CONVS = 3
        self.CNN_IN_C = [1, 32, 64]
        self.CNN_OUT_C = [32, 64, 128]
        self.CNN_CONV_KERNEL_SIZE = [3, 3, 3]
        self.CNN_MAX_KERNEL_SIZE = [4, 4, 4]
        ## DNN
        self.DNN_IN = [self.CNN_OUT_LENGTH+self.LYRICS_VEC_LENGTH+self.ARTIST_VEC_LENGTH, 256, 128]
        self.DNN_HIDDEN = [256, 128, 64]
        self.DNN_NUM_HIDDENS = len(self.DNN_HIDDEN)

        # 路径
        self.W2V_PATH = "/Users/inkding/Desktop/netease2/models/w2v/c4.mod"
        self.D2V_PATH = "/Users/inkding/Desktop/netease2/models/d2v/d2v_a2.mod"    
        self.ARTISTS_VEC_DICT_PATH = "/Users/inkding/Desktop/netease2/data/artists_vec_dict.pkl"


if __name__ == '__main__':
    print(Config().__dict__)