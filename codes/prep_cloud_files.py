import os
import sys
import shutil

'''
准备云服务器上需要的数据。
+ 音频：直接上传 vggish_examples 文件夹里的所有数据（1.64G）
+ 歌词：将数据集中的歌词文件整理出来（21M）上传
+ 评论：直接上传正负样本的 json 文件即可【可以直接从github上下载】
+ 歌手：上传 artist 字典文件【可以直接从github上下载】
'''

def prep_lyrics_files():
	# 准备歌词文件
	# 数据集
	pos_tracks = open("../data_related/tracks/pos_tracks_cls_vgg.txt").read().split()
	neg_tracks = open("../data_related/tracks/neg_tracks_cls_vgg.txt").read().split()
	tracks = pos_tracks + neg_tracks

	save_dir = "../data/cloud_files/data/lyrics_json"
	file_count = 0
	for root, dirs, files in os.walk("/Volumes/nmusic/NetEase2020/data/proxied_lyrics"):
		for file in files:
			if file[:-5] in tracks:
				save_dir_ = "{}/{}".format(save_dir, file_count//100) # 二级文件夹容量为100
				if not os.path.exists(save_dir_):
					os.makedirs(save_dir_)
				src = os.path.join(root, file)
				dst = os.path.join(save_dir_, file)
				shutil.copy(src, dst) # 复制文件
				file_count += 1


if __name__ == '__main__':
	prep_lyrics_files()

