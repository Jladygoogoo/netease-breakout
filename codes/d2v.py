import logging

from gensim.models import Doc2Vec
from preprocess import TaggedSentenceGenerator, replace_noise, cut

def train_d2v(read_path, log_path):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(message)s", datefmt="%H:%M:%S", filename=log_path)

	logging.info("start training a new model - d2v_a2.mod...\n")
	model = Doc2Vec(documents=TaggedSentenceGenerator(read_path), 
					dm=1, vector_size=300, window=8, 
					workers=8, epochs=10)
	model.save("../models/d2v/d2v_a2.mod")

def get_doc_vector(text, model):
	# model = Doc2Vec.load(m_path)
	words = cut(replace_noise(text), join_en=False)
	vec = model.infer_vector(doc_words=words)

	return vec


if __name__ == '__main__':
	read_path = "/Volumes/nmusic/NetEase2020/data/proxied_lyrics"
	log_path = "../logs/d2v.log"
	train_d2v(read_path, log_path)
