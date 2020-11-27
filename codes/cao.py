import os
import json
from connect_db import MyConn


conn = MyConn()
# for root, dirs, files in os.walk("/Volumes/nmusic/NetEase2020/data/breakouts_rawmusic"):
# 	for file in files:
# 		if "DS" in file: continue
# 		track_id = file[:-4]
# 		path = os.path.join(root, file)
# 		try:
# 			conn.update(settings={"rawmusic_path": path}, conditions={"track_id": track_id})
# 		except:
# 			print("track - {} failed.".format(track_id))

have_rawmusic = [str(r[0]) for r in conn.query(sql="SELECT track_id FROM tracks WHERE rawmusic_path IS NOT NULL")]
# print(len(have_rawmusic))
breakouts = [r[0] for r in conn.query(sql="SELECT breakout_id FROM breakouts")]

for b in breakouts:
	if b.split('-')[0] in have_rawmusic:
		conn.update(table="breakouts", settings={"have_rawmusic":1}, conditions={"breakout_id": b})



# for b in breakouts:
# 	# print(breakout_id)
# 	# print(b["feature_words"])
# 	settings = {
# 		"breakout_id": b["track_id"] + '-' + str(b["flag"]),
# 		"feature_words": " ".join(b["feature_words"])
# 	}
# 	conn.insert(settings=settings, table="breakouts_feature_words_1")


