import os
import csv
from collections import defaultdict

from srl_evaluator import get_dataset, evaluate, evaluate_keysent, get_pred_key_sent


def is_match(key_sent, original_sent):
    original_match_string = ''.join(original_sent.split())
    for line in key_sent.strip().split('\n'):
        line = line.strip()
        if "SENT: " in line:
            sent = line.split('SENT: ')[1]
            match_string = ''.join(sent.split())
            if match_string == original_match_string:
                return True
    return False

if __name__ == '__main__':
    gt, sents, gt_verbs, gt_args = get_dataset()
    sota_pred_keysent = get_pred_key_sent()

    kiddon_output_dir = "../RecipeInterpretation/data/select-chunked"
    pred_useful = defaultdict(list)
    pred_verbs = defaultdict(list)
    pred_args = defaultdict(list)

    for youtube_id in sents:
        with open(os.path.join(kiddon_output_dir, youtube_id+'.txt'), encoding='utf-8') as cf:
            key_sents = cf.read().split('\n\n')
        for original_sent in sents[youtube_id]:
            match_found = False
            for key_sent in key_sents:
                sent = None
                verbs = []
                args = []
                if is_match(key_sent, original_sent):
                    match_found = True
                    for line in key_sent.strip().split('\n'):
                        line = line.strip()
                        if "PRED: " in line:
                            verbs.append(line.split('PRED: ')[1])
                        elif "DOBJ: " in line:
                            args.append(line.split('DOBJ: ')[1])
                        elif "PARG: " in line:
                            args.append(line.split('PARG: ')[1])
                    pred_verbs[youtube_id].append(verbs)
                    pred_args[youtube_id].append(args)
                    pred_useful[youtube_id].append(1)
                    break
            if not match_found:
                pred_verbs[youtube_id].append([])
                pred_args[youtube_id].append([])
                pred_useful[youtube_id].append(0)

    print("Key sentence:")
    evaluate_keysent(gt, pred_useful)

    print("Verbs:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_verbs, pred_verbs, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)

    print("Arguments:")
    for sota_pred in (None, sota_pred_keysent):
        for fuzzy, partial_ratio in ((False, False), (True, False), (True, True)):
            evaluate(gt_args, pred_args, fuzzy=fuzzy, partial_ratio=partial_ratio, sota_pred=sota_pred)
