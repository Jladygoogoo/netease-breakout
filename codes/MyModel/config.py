class Config:
    json_path = "/Users/inkding/Desktop/netease2/data/breakouts-u2.json"
    # json_path = "/Users/inkding/Desktop/netease2/data/test/breakouts-test0.json"
    w2v_path = "/Users/inkding/Desktop/partial_netease/models/word2vec/b1.mod"
    d2v_path = "/Users/inkding/Desktop/netease2/models/d2v/d2v_a1.mod"
    batch_size = 32
    topk = 6
    lr = 0.05
    num_epochs = 6
    log_step = 10
    save_epoch = 3
    embedding_size = 200
    cnn_out_size = 1024