import os
import shutil
import csv

path = "/s1_md0/v-fanxu/raw_videos_all"


youtube_ids = set()
with open("../youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f:
    reader = csv.DictReader(gt_f)
    for row in reader:
        youtube_id = row['VideoUrl'].split('?v=')[1]
        youtube_ids.add(youtube_id)

print(len(youtube_ids))
files = {}
for yid in youtube_ids:
    files[yid] = "NULL"
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        for yid in youtube_ids:
            if yid in file:
                files[yid] = (os.path.join(r, file))

target = "/s1_md0/v-fanxu/Extraction/youcook2/videos"
for k in files:
    print(k, files[k])
    shutil.copy(files[k], target)
