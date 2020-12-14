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

jieba.load_userdict("/Users/inkding/Desktop/netease2/resources/grams_0.txt")

# ============ #
# = 评论预处理 = #
# ============ #

def replace_noise(text):
	# 除去标点符号
	# 注意颜文字中的特殊符号
	puncs = open("/Users/inkding/Desktop/netease2/resources/punctuations.txt").read().splitlines()
	for p in puncs:
		text = text.lower().replace(p, '')

	# 去除emojis
	re_emoji = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]',re.UNICODE)
	text = re.sub(re_emoji,'',text)

	return text

# 给2grams切词
def raw_cut(text, min_size = 2):
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
def cut(text, join_en=True, deep_cut=False):

	words = []
	# 处理英文
	# 不考虑中英文先后顺序
	if join_en:
		for en_ws in re.findall(r'[\u4e00-\u9fa5 ]*([a-zA-z ]+)[\u4e00-\u9fa5 ]*' ,text):
			en_w = '-'.join(en_ws.split())
			if len(set(en_w))>1:
				words.append(en_w)
		text = re.sub(r'[a-zA-z ]', '', text)

	# 处理中文
	stops = open("/Users/inkding/Desktop/netease2/resources/stopwords.txt").read().splitlines()
	for cn_w in jieba.cut(text):
		if len(cn_w)>=2 and cn_w not in stops:
			words.append(cn_w)

	return words


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

				yield models.doc2vec.TaggedDocument(words,[str(flag)])

				flag += 1
				if flag%100==0:
					print("load {} files in total.".format(flag))


# 提取句子中的top_tags
def tags_extractor(text, topk=8, w2v_model=None):
	text = replace_noise(text)

	words = []
	for line in text.splitlines():
		words.extend(cut(line))
	counter = Counter(words)

	tags = []
	if w2v_model is not None:
		for x in counter.most_common():
			if not w2v_model.wv.__contains__(x[0]): continue
			tags.append(x[0])
			if len(tags)==topk:
				return tags
	else:
		tags = [x[0] for x in counter.most_common(topk)]
		return tags


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
	text = "日推第一！终于给我推人声了。"
	print(tags_extractor(text))
