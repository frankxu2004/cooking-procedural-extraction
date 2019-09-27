import csv
import json

youtube_ids = set()
with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f:
    reader = csv.DictReader(gt_f)
    for row in reader:
        youtube_id = row['VideoUrl'].split('?v=')[1]
        youtube_ids.add(youtube_id)

with open("youcook2/youtube_ids.txt", 'w', encoding='utf-8') as id_f:
    for idx in youtube_ids:
        id_f.write(idx+"\n")

total_ids = len(youtube_ids)
print("Total vids: ", total_ids)
val_count = 0
with open("youcook2/yc2_bb/yc2_bb_val_annotations.json", encoding="utf-8") as val_f:
    vals = json.load(val_f)
    for k in vals["database"]:
        if k in youtube_ids:
            val_count += 1

print("Val in ours: ", val_count)

test_count = 0
with open("youcook2/yc2_bb/yc2_bb_public_test_annotations.json", encoding="utf-8") as test_f:
    tests = json.load(test_f)
    for k in tests["database"]:
        if k in youtube_ids:
            test_count += 1

print("Test in ours: ", test_count)
