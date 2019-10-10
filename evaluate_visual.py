import json
import os
import csv
from collections import defaultdict

from srl_evaluator import get_dataset, evaluate, get_pred_key_sent

def merge_pred(pred_a, pred_b, gold):
    merged_pred = defaultdict(list)
    for yid in pred_a:
        assert len(pred_a[yid]) == len(pred_b[yid]) == len(gold[yid])
        for a, b, g in zip(pred_a[yid], pred_b[yid], gold[yid]):
            merged = list(set(a).intersection(set(g)).union(set(b)))
            merged_pred[yid].append(merged)
    return merged_pred

if __name__ == '__main__':
    gt, sents, gt_verbs, gt_args = get_dataset()
    sota_pred_keysent = get_pred_key_sent()
    srl_pred_verbs, srl_pred_args = json.load(open("youcook2/srl_pred.json", encoding='utf-8'))

    pred_verbs = defaultdict(list)
    pred_args = defaultdict(list)

    with open("youcook2/reviewed_0812_vid_5s.csv") as gt_f:
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            video_output = row['Video Pred'].strip().split(", ")
            pred_verbs[youtube_id].append([x.split()[0] for x in video_output if x])
            pred_args[youtube_id].append([x.split()[1] for x in video_output if x])

    print("Video only")
    print("Verbs:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_verbs, pred_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Arguments:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_args, pred_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Fusion!!!!")
    merged_verbs = merge_pred(pred_verbs, srl_pred_verbs, gt_verbs)
    merged_args = merge_pred(pred_args, srl_pred_args, gt_verbs)
    print("Verbs:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_verbs, merged_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Arguments:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_args, merged_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)
