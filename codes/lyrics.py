import os
import re
import json
import jieba
import enchant

from connect_db import MyConn
from preprocess import replace_noise, cut_en


def mark_language():
	'''
	对歌词库中的所有歌曲进行语种标记。
	'''
	conn = MyConn()
	enchant_dict = enchant.Dict("en_US")
	for track_id, lyrics_path in conn.query(sql="SELECT track_id, lyrics_path FROM tracks WHERE lyrics_path is not null"):
		with open(lyrics_path) as f:
			content = json.load(f)
		lyrics = replace_noise(content["lrc"]["lyric"])
		lyrics = re.sub(r"( )*[作词|作曲|编曲|制作人|录音|混母带|监制].*\n", "", lyrics)
		if len(lyrics)<10: # 说明没有东西
			language = "empty"
		language = _mark_language(lyrics, enchant_dict)
		conn.update(table="tracks", settings={"language":language}, conditions={"track_id":track_id})



def _mark_language(text, enchant_dict=None):
	'''
	mark_language的具体文本操作。
	'''
	text = text.lower()
	ch_text = re.findall(r"[\u4e00-\u9fa5]", text)
	count_not_en_seq = 0
	if len(ch_text)<15: # 说明很有可能是外文歌曲
		words = cut_en(" ".join(re.findall(r"[a-z ]+", text))) # 这里没有对停词进行去除
		for w in words:
			if len(w)==0: continue
			if not enchant_dict.check(w): 
				count_not_en_seq += 1
			else:
				count_not_en_seq = 0
			if count_not_en_seq==10:
				print(words[:20])
				return "else"
		return "en"
	return "ch"






if __name__ == '__main__':
	mark_language()
	# for file in os.listdir("/Volumes/nmusic/NetEase2020/data/proxied_lyrics/5/97"):
	# 	if "OS" in file:
	# 		continue
	# 	filepath = os.path.join("/Volumes/nmusic/NetEase2020/data/proxied_lyrics/5/97", file)
	# 	with open(filepath) as f:
	# 		content = json.load(f)	
	# 	lyrics = replace_noise(content["lrc"]["lyric"])
	# 	lyrics = re.sub(r"( )*[作词|作曲|编曲|制作人|录音|混母带|监制].*\n", "", lyrics)
	# 	print(lyrics)
