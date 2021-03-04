

def get_X(track_id, use_mp3, use_lyrics, use_artist, use_tags,
        ar_d2v_model, lyrics_d2v_model, ar_KB_df, ar_sup_json, name_2_KB_ents):
    conn = MyConn()
    with open("../data/artists_vec_dict.pkl", "rb") as f:
        d_artist_vec = pickle.load(f)

    # rawmusic_path, lyrics_path, artist, artist_text_src, tags = conn.query(
    #     table="sub_tracks", conditions={"track_id":track_id}, fetchall=False,
        # targets=["rawmusic_path", "lyrics_path", "artist", "artist_text_src", "tags"])

    vecs = []
    if use_mp3:
        mfcc_vec = get_mfcc(rawmusic_path).ravel()
        vecs.append(mfcc_vec)
    if use_lyrics:
        lyrics_vec = get_d2v_vector(lyrics_path, lyrics_d2v_model)
        vecs.append(lyrics_vec)
    if use_artist:
        # artist = artist.lower().strip()
        # if artist_text_src==1: # 在知识库中有对应艺人的信息
        #     text = str(generate_description_KB(artist, ar_KB_df, name_2_KB_ents, mode=3))
        # else: # 有网易云音乐补充对应艺人信息
        #     text = str(generate_description_sup(artist, ar_sup_json))
        # artist_vec = ar_d2v_model.infer_vector(cut(text))
        vecs.append(artist_vec)
    if use_tags:
        tags_vec = get_tags_vector(tags.split())
        vecs.append(tags_vec)
    features_vec = concatenate_features(vecs)

    return features_vec


def build_dataset():
    conn = MyConn()
    dataset_size = 600
    pos_tracks = conn.query(sql="SELECT track_id FROM sub_tracks WHERE valid_bnum>0 LIMIT {}".format(dataset_size))
    neg_tracks = conn.query(sql="SELECT track_id FROM sub_tracks WHERE valid_bnum=0 LIMIT {}".format(dataset_size))
    lyrics_d2v_model = Doc2Vec.load("../models/d2v/d2v_a2.mod") # 歌词d2v模型
    # ar_d2v_model = Doc2Vec.load("../models/d2v/ar_m3w5.mod") # 艺人信息d2v模型
    # ar_KB_df = pd.read_csv("../data/artists_KB.csv") # 知识库艺人信息
    # with open("../data/sup_artists_desc_2.json") as f: # 网易云音乐补充艺人信息
    #     ar_sup_json = json.load(f)
    with open("../data/name_2_KB_ents.pkl", "rb") as f:
        name_2_KB_ents = pickle.load(f)

    X, y = [], []
    args = {
        "lyrics_d2v_model": lyrics_d2v_model,
        "ar_d2v_model": ar_d2v_model,
        "ar_KB_df": ar_KB_df,
        "ar_sup_json": ar_sup_json,
        "name_2_KB_ents": name_2_KB_ents,
        "use_mp3": True,
        "use_lyrics": True,
        "use_artist": True,
        "use_tags": False
    }

    def add_data(tracks, label):
        for t in tracks:
            try:
                X.append(get_X(track_id=t, **args))
                y.append(label)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except:
                print(label, t)
                print(traceback.format_exc())
    add_data(pos_tracks, 1)
    add_data(neg_tracks, 0)

    dataset_name = "m"*args["use_mp3"] + "l"*args["use_lyrics"] + "a"*args["use_artist"] + "t"*args["use_tags"]\
                    + str(len(pos_tracks))
    with open("../data/dataset/{}.pkl".format(dataset_name), 'wb') as f:
        pickle.dump([X,y], f)



class TaggedSentenceGenerator():
    def __init__(self,path,mode='train'):
        self.path = path
        self.mode = mode

    def __iter__(self):
        flag = 0
        tag2track = {}
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if "DS" in file: continue
                with open(os.path.join(root, file)) as f:
                    content = json.load(f)
                if "lrc" not in content or "lyric" not in content["lrc"]:
                    continue
                text = replace_noise(content["lrc"]["lyric"])
                words = cut(text, join_en=False)
                if len(words)<10:
                    continue

                yield models.doc2vec.TaggedDocument(words,[str(flag)])

                flag += 1
                if flag%100==0:
                    print("load {} files in total.".format(flag))
