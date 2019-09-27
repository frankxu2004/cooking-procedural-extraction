import datetime
import json
import random
import csv


random.seed(1)

with open('youcook2/3.video_item_list.with_transcript.azure_stt.youtube_auto_generated.json', encoding='utf-8') as f:
    dataset = json.load(f)

# filter out data without transcripts
filtered_dataset = []
for d in dataset:
    if d['transcript']:
        filtered_dataset.append(d)

with open('youcook2/3.video_item_list.with_transcript.azure_stt.youtube_auto_generated_filtered.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_dataset, f, indent=4)

id2label = {}
with open('youcook2/label_foodtype.csv', encoding='utf-8') as typef:
    for line in typef:
        line = line.strip()
        id2label[line.split(',')[0]] = line.split(',')[1]

print(len(filtered_dataset))

type2data = {}

for data in filtered_dataset:
    recipe_type = data['recipe_type']
    if recipe_type not in type2data:
        type2data[recipe_type] = [data]
    else:
        type2data[recipe_type].append(data)

print(len(type2data))

with open('youcook2/tolabel_azure.csv', 'w', newline='', encoding='utf-8') as outfile:
    csvwriter = csv.writer(outfile)
    for recipe_type in type2data:
        recipe_label = id2label[recipe_type]
        to_label = random.sample(type2data[recipe_type], 4)
        for item in to_label:
            url = item['url']
            d = item['transcript']['en']['azure_stt_transcripts']['sentence_level']
            for idx, sent in enumerate(d):
                sent_text = sent['text']
                sent_start = str(datetime.timedelta(milliseconds=int(sent['segment'][0])))
                csvwriter.writerow([recipe_label, url, sent_start, sent_text, str(idx)])

