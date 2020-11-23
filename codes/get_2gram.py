#coding:utf-8
import re
import os
import sys
import json
import pickle
import numpy as np 

import jieba
from collections import Counter
from preprocess import replace_noise, raw_cut

gram_skip = open("../resources/gram_skip.txt").read().splitlines()

def get_all_2grams(text):
	slices = []
	words = raw_cut(text, min_size=1)

	for i in range(len(words)-1):
		if not re.search('[\u4e00-\u9fa5]',''.join(words[i:i+2])): continue
		if words[i]==words[i+1]: continue
		if words[i] in gram_skip or words[i+1] in gram_skip: continue

		gram = ''.join(words[i:i+2])
		if len(gram)>2 and len(gram)<5:
			slices.append(gram)

	return slices


# textÖÐ°üº¬»»ÐÐ
def get_top_2grams(text, top_rate=0.02, minn=30):
	grams = []
	top_grams = []

	text = replace_noise(text)
	for line in text.splitlines()[:1000]:
		grams.extend(get_all_2grams(line))
	counter = Counter(grams)
	topk = min(int(len(grams)*top_rate), 10)
	for k,v in counter.most_common(topk):
		if v>=minn: 
			# print(k)
			top_grams.append(k)

	return top_grams



def run(read_path, write_path):
	grams_set = set(open(write_path).read().splitlines())
	f_count = 0

	for root,dirs,files in os.walk(read_path):
		for file in files:
			f_count += 1
			if f_count<2600: continue

			if 'DS' in file: continue
			text = open(os.path.join(root, file)).read()
			for gram in get_top_2grams(text):
				grams_set.add(gram)

			if f_count % 100 == 0:
				print("read %d file, grams_set size is %d" % (f_count, len(grams_set)))

				with open(write_path, 'w') as f:
					f.write('\n'.join(grams_set))



def test(path):
	flag = 0
	for root,dirs,files in os.walk(path):
		for file in files:
			if 'DS' in file: continue
			text = open(os.path.join(root, file)).read()
			# text = open(path).read()
			get_top_2grams(text)



if __name__ == '__main__':
	# main run
	read_path = '/Volumes/nmusic/NetEase2020/data/simple_proxied_reviews_text'
	write_path = '../resources/2grams.txt'
	run(read_path, write_path) # 3900

	# test
	path = "/Users/inkding/Desktop/partial_netease/data/proxied_breakouts_text/0/34"
	# test(path)
