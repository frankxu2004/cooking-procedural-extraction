import shutil
import os
import csv
from collections import defaultdict

source = "../RecipeInterpretation/data/transcript"
target = "../RecipeInterpretation/data/select"

transcripts = defaultdict(list)
with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f:
    reader = csv.DictReader(gt_f)
    for row in reader:
        youtube_id = row['VideoUrl'].split('?v=')[1]
        sentence = row['Sentence']
        transcripts[youtube_id].append(sentence)

for youtube_id in transcripts:
    with open(os.path.join(target, youtube_id+'.txt'), 'w', encoding='utf-8') as wf:
        wf.write("\n".join(transcripts[youtube_id]))
