class Config:
    breakouts_id_train = "/Users/inkding/Desktop/netease2/data/dataset/breakouts_id_train_1.txt"
    no_breakouts_id_train = "/Users/inkding/Desktop/netease2/data/dataset/no_breakouts_id_train_1.txt"
    breakouts_id_test = "/Users/inkding/Desktop/netease2/data/dataset/breakouts_id_test_1.txt"
    no_breakouts_id_test = "/Users/inkding/Desktop/netease2/data/dataset/no_breakouts_id_test_1.txt"

    w2v_path = "/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod"
    d2v_path = "/Users/inkding/Desktop/netease2/models/d2v/d2v_a1.mod"
    batch_size = 32
    topk = 6
    lr = 0.05
    num_epochs = 5
    log_step = 10
    save_epoch = 1
    embedding_size = 300
    cnn_out_size = 1024
    class_num = 2
    breakout_size = 1000
    no_breakout_size = 1000