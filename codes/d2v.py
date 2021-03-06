import re
import json
import logging

from gensim.models import Doc2Vec
from preprocess import TaggedSentenceGenerator, replace_noise, cut
from connect_db import MyConn

def train_d2v(log_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(message)s", datefmt="%H:%M:%S", filename=log_path)

    model = Doc2Vec(documents=TaggedSentenceGenerator(), 
                    dm=1, vector_size=300, window=8, 
                    workers=8, epochs=10)
    model.save("../models/d2v/d2v_b1.mod")



def get_doc_vector(text, model):
    # model = Doc2Vec.load(m_path)
    words = cut(replace_noise(text), join_en=False)
    vec = model.infer_vector(doc_words=words)

    return vec


def test_d2v_with_source(text, model, topn=5):
    conn = MyConn()
    source_tracks = open("../data_related/lyrics_valid_tracks.txt").read().splitlines()
    text = replace_noise(text)
    text = re.sub(r"( )*[作词|作曲|编曲|制作人|录音|混母带|监制].*\n", "", text)
    words = cut(text, join_en=False)
    vec = model.infer_vector(words)
    s = model.docvecs.most_similar([vec], topn=10)
    print(s)
    # for index, score in s:
    #     track_id = source_tracks[int(index)]
    #     print(index, score, track_id)
    #     lyrics_path = conn.query(targets=["lyrics_path"], conditions={"track_id":track_id}, fetchall=False)[0]
    #     with open(lyrics_path) as f:
    #         print(replace_noise(json.load(f)["lrc"]["lyric"]), "\n")



def test_d2v_unseen_docs(text1, text2, model):
    def preprocess(text):
        text = re.sub(r"( )*[作词|作曲|编曲|制作人|录音|混母带|监制].*\n", "", replace_noise(text))
        return cut(text, join_en=False)
    words1 = preprocess(text1)
    words2 = preprocess(text2)
    print("simi score: {:.3f}".format(
        model.docvecs.similarity_unseen_docs(model, words1, words2)))



if __name__ == '__main__':
    # read_path = "/Volumes/nmusic/NetEase2020/data/proxied_lyrics"
    # log_path = "../logs/d2v.log"
    # train_d2v(log_path)
    # generator = TaggedSentenceGenerator()
    # for r in generator:
    #     pass
    # generator.save_lyrics_valid_tracks()

    filepath1 = "/Volumes/nmusic/NetEase2020/data/proxied_lyrics/0/5/1306507078.json"
    with open(filepath1) as f:
        text1 = json.load(f)["lrc"]["lyric"]
    filepath2 = "/Volumes/nmusic/NetEase2020/data/proxied_lyrics/0/55/1296896326.json"
    with open(filepath1) as f:
        text2 = json.load(f)["lrc"]["lyric"]
    # text1 = open("../resources/lyrics_text1.txt").read()
    # text2 = open("../resources/lyrics_text2.txt").read()
    model = Doc2Vec.load("../models/d2v/d2v_a2.mod")

    # test_d2v_with_source(text1, model)
    test_d2v_unseen_docs(text1, text2, model)

