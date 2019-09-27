import os
import csv
from collections import defaultdict

from srl_evaluator import get_dataset

gt, sents, gt_verbs, gt_args = get_dataset()

kiddon_output_dir = "../RecipeInterpretation/data/select-chunked"

for youtube_id in sents:
    with open(os.path.join(kiddon_output_dir, youtube_id+'.txt'), encoding='utf-8') as cf:
        key_sents = cf.read().split('\n\n\n')
        print(key_sents[8].split('\n'))
        break
