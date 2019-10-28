import csv
import json
from collections import defaultdict

from fuzzywuzzy import fuzz

from srl_evaluator import get_dataset, evaluate, get_pred_key_sent


def read_epic():
    epic_verbs = {}
    with open('youcook2/EPIC_verb_classes.csv', encoding='utf-8') as epic_f:
        reader = csv.DictReader(epic_f)
        for row in reader:
            key = row['class_key']
            words = [w.replace('-', ' ') for w in eval(row['verbs'])]
            epic_verbs[key] = list(set(words))

    epic_nouns = {}
    with open('youcook2/EPIC_noun_classes.csv', encoding='utf-8') as epic_f:
        reader = csv.DictReader(epic_f)
        for row in reader:
            key = row['class_key']
            words = [" ".join(w.split(':')[::-1]) for w in eval(row['nouns'])]
            epic_nouns[key] = list(set(words))
    return epic_verbs, epic_nouns

def merge_pred_oracle(pred_a, pred_b, gold, epic):
    merged_pred = defaultdict(list)
    for yid in pred_a:
        assert len(pred_a[yid]) == len(pred_b[yid]) == len(gold[yid])
        for a, b, g in zip(pred_a[yid], pred_b[yid], gold[yid]):
            expanded_a = []
            for x in a:
                expanded_a += epic[x]
            merged_pred[yid].append(set(b).union(set(expanded_a).intersection(set(g))))
    return merged_pred


def merge_pred(pred_a, pred_b, gold, epic):
    merged_pred = defaultdict(list)
    for yid in pred_a:
        assert len(pred_a[yid]) == len(pred_b[yid]) == len(gold[yid])
        for a, b, g in zip(pred_a[yid], pred_b[yid], gold[yid]):
            merged = list(set(a).union(set(b)))
            merged_pred[yid].append(merged)
    return merged_pred

def select_pred(pred, gold, epic):
    selected_pred = defaultdict(list)
    for yid in pred:
        assert len(pred[yid]) == len(gold[yid])
        for a, g in zip(pred[yid], gold[yid]):
            best_candidates = []
            for x in a:
                max_score = 0
                best_candidate = None
                for candidate in epic[x]:
                    for t in g:
                        score = fuzz.partial_ratio(candidate, t)
                        if max_score < score:
                            max_score = score
                            best_candidate = candidate
                if best_candidate:
                    best_candidates.append(best_candidate)
                else:
                    best_candidates.append(x)
            assert None not in best_candidates
            selected_pred[yid].append(best_candidates)
    return selected_pred

if __name__ == '__main__':
    gt, sents, gt_verbs, gt_args = get_dataset()
    sota_pred_keysent = get_pred_key_sent()
    srl_pred_verbs, srl_pred_args = json.load(open("youcook2/srl_pred.json", encoding='utf-8'))
    epic_verbs, epic_nouns = read_epic()

    pred_verbs = defaultdict(list)
    pred_args = defaultdict(list)

    with open("youcook2/reviewed_0812_vid_5s.csv") as gt_f:
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            video_output = row['Video Pred'].strip().split(", ")
            pred_verbs[youtube_id].append(list(set([x.split()[0] for x in video_output if x])))
            pred_args[youtube_id].append(list(set([x.split()[1] for x in video_output if x])))

    selected_pred_verbs = select_pred(pred_verbs, gt_verbs, epic_verbs)
    selected_pred_args = select_pred(pred_args, gt_args, epic_nouns)
    # print("Video only")
    # print("Verbs:")
    # for sota_pred in (None, sota_pred_keysent):
    #     for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
    #         evaluate(gt_verbs, selected_pred_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)
    #
    # print("Arguments:")
    # for sota_pred in (None, sota_pred_keysent):
    #     for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
    #         evaluate(gt_args, selected_pred_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    merge_pred_verbs = merge_pred(selected_pred_verbs, srl_pred_verbs, gt_verbs, epic_verbs)
    merge_pred_args = merge_pred(selected_pred_args, srl_pred_args, gt_args, epic_nouns)

    print("Fusion!!!!")
    print("Verbs:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_verbs, merge_pred_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Arguments:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_args, merge_pred_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Fusion Oracle!!!!")
    merged_verbs = merge_pred_oracle(pred_verbs, srl_pred_verbs, gt_verbs, epic_verbs)
    merged_args = merge_pred_oracle(pred_args, srl_pred_args, gt_verbs, epic_nouns)
    print("Verbs:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_verbs, merged_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Arguments:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_args, merged_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)
