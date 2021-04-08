import os
import jieba
import re
import json
from tqdm import tqdm
import pickle
import random
import numpy as np 
from collections import Counter 
from gensim import models
import pymysql

from connect_db import MyConn

jieba.load_userdict("/Users/inkding/Desktop/netease2/resources/grams_0.txt")

# ============ #
# = 评论预处理 = #
# ============ #

def replace_noise(text):
	# 除去[*]里面的表情
	text = re.sub(r"\[.+\]", '', text)

	# 除去标点符号
	# 注意颜文字中的特殊符号
	puncs = open("/Users/inkding/Desktop/netease2/resources/punctuations.txt").read().splitlines()
	for p in puncs:
		text = text.lower().replace(p, '')

	# 去除emojis
	re_emoji = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]',re.UNICODE)
	text = re.sub(re_emoji,'',text)

	return text


def raw_cut(text, min_size = 2):
	# 给2grams切词
	words = []
	# 处理英文
	# 不考虑中英文先后顺序
	for en_ws in re.findall(r'[\u4e00-\u9fa5 ]*([a-zA-z ]+)[\u4e00-\u9fa5 ]*' ,text):
		en_w = '-'.join(en_ws.split())
		if len(en_w)>1:
			words.append(en_w)
	text = re.sub(r'[a-zA-z]', '', text)

	# 处理中文
	stops = open("/Users/inkding/Desktop/netease2/resources/stopwords.txt").read().splitlines()
	for cn_w in jieba.cut(text):
		if len(cn_w)>=min_size and cn_w not in stops:
			words.append(cn_w)

	# print(words)
	return words


# 基本切词
def cut(text, join_en=True, stops_sup=None, filter_number=False):
	'''
	中英文分别处理（不考虑中英文先后顺序）
	连续英文用'-'连接，在切词时作为一个词
	'''
	words = []
	stops = open("/Users/inkding/Desktop/netease2/resources/stopwords.txt").read().splitlines()
	if stops_sup:
		stops = stops + stops_sup
	# 处理英文
	if join_en:
		for en_ws in re.findall(r'[\u4e00-\u9fa5 ]*([a-zA-z ]+)[\u4e00-\u9fa5 ]*' ,text):
			en_w = '-'.join(en_ws.split())
			if len(set(en_w))>1 and en_w not in stops:
				words.append(en_w)
		text = re.sub(r'[a-zA-z]{2,}', '', text)

	# 处理中文
	for cn_w in jieba.cut(text):
		if filter_number and re.match(r"(\d+日?|第.+)", cn_w): 
			continue
		if len(cn_w)>=2 and cn_w not in stops:
			words.append(cn_w)

	return words



def cut_en(text, stops_sup=[]):
	'''
	对英文进行切词
	'''
	stops = open("/Users/inkding/Desktop/netease2/resources/stopwords.txt").read().splitlines() + stops_sup
	clean_words = []
	words = text.lower().split()
	for w in words:
		if w not in stops and len(w)>1:
			clean_words.append(w)
	return clean_words



# 为w2v模型封装生成器
class W2VSentenceGenerator():
	def __init__(self,path, min_size=2, file_is_sent=True):
		self.path = path
		self.min_size = min_size
		self.file_is_sent = file_is_sent

	def __iter__(self):
		for root, dirs, files in os.walk(self.path):
			for file in files:
				if not '.txt' in file: continue
				reviews = open(os.path.join(root,file)).read().splitlines()
				# 每首歌的评论最多选取 1000 条（采样）
				text = replace_noise("\n".join(random.sample(reviews, min(1000, len(reviews)))))

				# 将整个文档看作一个句子
				if self.file_is_sent:
					file_sent = []
					for line in text.splitlines():
						file_sent.extend(cut(line))
					yield file_sent

				# 读取文档中的每一行为一个句子
				else:
					for line in text.splitlines():
						sent = cut(line)
						if len(sent)>=self.min_size:
							yield sent

# 为doc2vec模型封装生成器（加上文档标签和数目限制）
# 歌词数据
class TaggedSentenceGenerator():
	'''
	注意：中文歌词好说，正常分词即可，外文歌词注意只保留英文（非西语）
	数据源：数据库（文件路径中并不是所有的歌词都可以用作训练）
	'''
	def __init__(self):
		sql = "SELECT track_id, lyrics_path FROM tracks WHERE language IN ('en','ch')"
		self.lyrics_path_set = MyConn().query(sql=sql)

	def __iter__(self):
		flag = 0
		self.lyrics_valid_tracks = [] # 用于训练歌词d2v模型的歌曲id
		for i, p in enumerate(self.lyrics_path_set):
			track_id, lyrics_path = p
			if i%1000==0: print("{} files loaded.".format(i))

			with open(lyrics_path) as f:
				content = json.load(f)
				if "lrc" not in content or "lyric" not in content["lrc"]:
					continue
				text = replace_noise(content["lrc"]["lyric"])
				text = re.sub(r"( )*[作词|作曲|编曲|制作人|录音|混母带|监制].*\n", "", text)
				words = cut(text, join_en=False)
				if len(words)<10:
					continue
				self.lyrics_valid_tracks.append(track_id)

				yield models.doc2vec.TaggedDocument(words,[str(flag)])

	def save_lyrics_valid_tracks(self):
		with open("../data_related/lyrics_valid_tracks.txt", "w") as f:
			f.write("\n".join(self.lyrics_valid_tracks))



# 提取句子中的top_tags
def tags_extractor(text, topk=10, w2v_model=None, stops_sup=None, return_freq=False):
	text = replace_noise(text)

	words = []
	for line in text.splitlines():
		words.extend(cut(line, stops_sup=stops_sup))
	counter = Counter(words)

	tags = []
	# rubbish = open("../resources/rubbish_tags.txt").read().splitlines()
	if w2v_model is not None:
		for x in counter.most_common():
			if not w2v_model.wv.__contains__(x[0]): continue
			# if x[0] in rubbish: continue
			tags.append(x)
			if topk and len(tags)==topk:
				break
	else:
		tags = counter.most_common(topk)
	d_tag_freq = dict(tags)

	if not return_freq:
		return list(d_tag_freq.keys())
	else:
		total = sum(d_tag_freq.values())
		return list(map(lambda p:(p[0], p[1]/total), list(d_tag_freq.items())))


# ============ #
# = 歌词预处理 = #
# ============ #

def extract_lyrics(path, timemark=False):
	# timemark=True 表示返回时间标记
	with open(path) as f:
		content = json.load(f)['lrc']['lyric']
		lines = content.split('\n')

	timemark_r = r'\[\d{2}:\d{2}[:.]*\d{0,3}\]'

	if re.search(timemark_r,content):
		tmark_text = []
		for l in lines:
			tmarks = re.findall(timemark_r,l)

			if len(tmarks)==0: continue
			text = re.search(r'\]([^\[\]]+)',l)
			if text:
				text = text.group(1).strip()
				for tmark in tmarks:
					tmark_text.append((tmark,text))
		tmark_text = sorted(tmark_text,key=lambda x:x[0])

		tmarks = [p[0] for p in tmark_text]
		lyrics = [p[1] for p in tmark_text]
	else:
		tmarks = None
		lyrics = lines

	if timemark:
		return tmarks,lyrics
	else:
		return list(filter(None,lyrics))


def extract_lyrics_as_files(tracks_set, write_dir):
	# 将tracks_set中所有歌曲的歌词提取出来写为文件，并保存在write_dir中
	# 连接数据库，可根据track_id快速定位歌词文件
	conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
							db="NetEase_proxied", password="SFpqwnj285798,.")
	cursor = conn.cursor()
	sql = "SELECT lyrics_path FROM tracks WHERE track_id=%s"

	for tid in tracks_set:
		cursor.execute(sql, (tid,))
		lyrics_path = "/Volumes/nmusic/NetEase2020/data" + cursor.fetchone()[0]
		lyrics = extract_lyrics(lyrics_path)
		with open(os.path.join(write_dir, "{}.txt".format(tid)), 'w') as f:
			f.write('\n'.join(lyrics))


if __name__ == "__main__":
	text = "把证据发给FBI啊[大哭][大哭][大哭]"
	print(cut(replace_noise(text)))
