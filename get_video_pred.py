import csv
from collections import defaultdict

clip_data = defaultdict(list)

with open("youcook2/clip_manifest_10s.txt", encoding='utf-8') as clip_f, \
        open("youcook2/clip_prediction_5s.txt", encoding='utf-8') as pred_f:
    for line in clip_f:
        ls = line.strip().split()
        youtube_id = ls[1]
        clip_id = ls[0]
        num = int(clip_id[-4:])
        start_time = float(ls[2])
        end_time = float(ls[3])
        if end_time - start_time > 5.0:
            clip_data[youtube_id].append([youtube_id +"_{0:0>4}".format(num*2-1), start_time, start_time + 5.0])
            clip_data[youtube_id].append([youtube_id +"_{0:0>4}".format(num*2), start_time + 5.0, end_time])
        else:
            clip_data[youtube_id].append([youtube_id +"_{0:0>4}".format(num*2-1), start_time, end_time])


    for line in pred_f:
        ls = line.strip().split()
        clip_id = ls[0].split('/')[1]
        pred_verb = ls[-2]
        pred_noun = ls[-1]
        youtube_id = clip_id[:11]
        for idx, clip in enumerate(clip_data[youtube_id]):
            if clip[0] == clip_id:
                clip_data[youtube_id][idx].append(pred_verb + " " + pred_noun)


with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f, \
        open("youcook2/reviewed_0812_vid_5s.csv", "w", newline='', encoding="utf-8") as out_f:
    fieldnames = ["No", "Title", "VideoUrl", "TimeStamp", "Sentence", "RowNumber", "IsUsefulSentence", "Key steps",
                  "Verb", "Object(directly related with Verb)", "Location", "Time", "Temperature",
                  "Other important phrase(like with", "Video Pred", "Clip IDs"]
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    reader = csv.DictReader(gt_f)
    rows = list(reader)
    for idx, row in enumerate(rows):
        youtube_id = row['VideoUrl'].split('?v=')[1]
        start_time = row['TimeStamp']
        sts = start_time.split(':')
        start_time = int(sts[0]) * 60 + float(sts[1])
        end_time = 1000000
        if idx+1 < len(rows):
            next_row = rows[idx+1]
            next_youtube_id = next_row['VideoUrl'].split('?v=')[1]
            if next_youtube_id == youtube_id:
                next_start_time = next_row['TimeStamp']
                next_sts = next_start_time.split(':')
                next_start_time = int(next_sts[0]) * 60 + float(next_sts[1])
                end_time = next_start_time

        vid_pred_res = []
        aligned_clip_ids = []
        clips = clip_data[youtube_id]
        for clip in clips:
            cst = clip[1]
            cet = clip[2]
            if (cst <= start_time <= cet) or (cst <= end_time <= cet) or (cst >= start_time and cet <= end_time):
                vid_pred_res.append(clip[3])
                aligned_clip_ids.append(clip[0])

        to_write = dict(row)
        to_write['Video Pred'] = ", ".join(vid_pred_res)
        to_write['Clip IDs'] = ", ".join(aligned_clip_ids)

        writer.writerow(to_write)
