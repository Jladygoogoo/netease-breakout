class Config:
    def __init__(self):
        # 数据
        self.LYRICS_EMBED_SIZE = 300
        self.ARTIST_EMBED_SIZE = 72
        self.WORD_EMBED_SIZE = 300
        self.TRAIN_SIZE = 2500 # pos:neg = 1:1
        self.VALID_SIZE = 300
        self.TEST_SIZE = 100
        self.BATCH_SIZE = 32
        self.NUM_CLASS = 2
        ## 音频特征提取
        self.SAMPLE_RATE=22050
        self.FFT_SIZE = 1024
        self.WIN_SIZE = 1024
        self.HOP_SIZE = 512
        self.NUM_MELS = 128
        self.NUM_MFCC = 20
        self.FEATURE_LENGTH = 1024
        self.DURATION = 10
        ## 评论数据
        self.TOPK = 5

        # 训练
        self.LEARNING_RATE = 1e-3
        self.STOPPING_RATE = 1e-6
        self.WEIGHT_DECAY = 1e-7
        self.MOMENTUM = 0.01
        self.FACTOR = 0.5
        self.PATIENCE = 5
        self.EPOCHS_NUM = 100
        self.EPOCH_SAVE_STEP = 5
        self.RANDOM_STATE = 21

        # 模型设置
        ## CUSTOM_CNN
        self.CUSTOMCNN_OUT_LENGTH = 2048
        self.CUSTOMCNN_NUM_CONVS = 3
        self.CUSTOMCNN_CHANNELS = [64, 128, 256]
        self.CUSTOMCNN_KERNEL_SIZES = [3, 3, 3]
        self.CUSTOMCNN_POOLING_SIZES = [2, 2, 2]
        ## DNN
        # self.DNNCLASSIFIER_HIDDENS = [2048, 1024, 512]
        self.DNNCLASSIFIER_DROPOUT_RATE = 0.01
        ## TEXT_CNN_EMBED
        self.EMBED_SIZE = 256
        self.TEXTCNN_KERNEL_SIZES = [] # 在run.py中重定义
        self.TEXTCNN_NUM_CHANNELS = 256
        self.TEXTCNN_DROPOUT_RATE = 0.01
        ## MUSIC_EMBED
        self.MUSICEMBED_HIDDENS = []
        self.MUSICEMBED_DROPOUT_RATE = 0.01
        

        # 路径
        self.W2V_PATH = "/Users/inkding/Desktop/netease2/models/w2v/c4.mod"
        self.D2V_PATH = "/Users/inkding/Desktop/netease2/models/d2v/d2v_b1.mod"    
        self.ARTISTS_VEC_DICT_PATH = "/Users/inkding/Desktop/netease2/data/artists_vec_dict/artists_vec_dict_r_minmax.pkl"
        self.REVIEWS_FEATURE_WORDS_WITH_FREQS_POS_PATH = "/Users/inkding/Desktop/netease2/data/reviews_feature_words_with_freqs/breakouts_cls.json"
        self.REVIEWS_FEATURE_WORDS_WITH_FREQS_NEG_PATH = "/Users/inkding/Desktop/netease2/data/reviews_feature_words_with_freqs/no_breakouts_cls.json"
        self.POS_TRACKS_FILEPATH = "/Users/inkding/Desktop/netease2/data_related/tracks/pos_tracks_cls.txt"
        self.NEG_TRACKS_FILEPATH = "/Users/inkding/Desktop/netease2/data_related/tracks/neg_tracks_cls.txt"
        self.D_POS_TRACK_BREAKOUT_PATH = "/Users/inkding/Desktop/netease2/data_related/tracks/d_pos_track_breakout_cls.pkl"


if __name__ == '__main__':
    print(Config().__dict__)