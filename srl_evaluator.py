import csv
import pickle
from collections import defaultdict

import nltk

from sklearn.metrics import classification_report
from utils import get_srl_predictor, read_vocab, parse_result, filter_chunks, get_args, get_verb
from fuzzywuzzy import fuzz

lemmatizer = nltk.stem.WordNetLemmatizer()

def filter_by_vocab(chunks, vocab):
    ret = []
    for chunk in chunks:
        for idx, c in enumerate(chunk):
            lemma = lemmatizer.lemmatize(c['text'], 'v')
            if c['type'] == "V" and lemma in vocab:
                chunk[idx]['text'] = lemma
                ret.append(chunk)
                break
    return ret


def process_gt_verbs(verb_str):
    if verb_str:
        v = []
        for p in verb_str.replace(";", ",").split(","):
            ps = p.strip().lower()
            if ps:
                # get root/first verb only
                v.append(ps.split()[0])
        return v
    else:
        return []


def process_annotation(string):
    if string:
        v = []
        for p in string.split(";"):
            ps = p.strip().lower()
            if ps:
                v.append(ps)
        return v
    else:
        return []


def write_predicted(pred_useful, pred_verbs, pred_args):
    with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f, \
            open("youcook2/reviewed_0812_pred.csv", "w", newline='', encoding="utf-8") as out_f:
        fieldnames = ["No", "Title", "VideoUrl", "TimeStamp", "Sentence", "RowNumber", "IsUsefulSentence", "Key steps",
                      "Verb", "Object(directly related with Verb)", "Location", "Time", "Temperature",
                      "Other important phrase(like with", "PredUseful", "PredVerbs", "PredArgs"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            row_num = int(row['RowNumber'])
            pred_u = pred_useful[youtube_id][row_num]
            pred_v = pred_verbs[youtube_id][row_num]
            pred_a = pred_args[youtube_id][row_num]
            to_write = dict(row)
            to_write['PredUseful'] = pred_u
            to_write['PredVerbs'] = pred_v
            to_write['PredArgs'] = pred_a
            writer.writerow(to_write)


def evaluate(gt, pred):
    total = 0
    exact_precs = 0.0
    exact_recalls = 0.0
    fuzzy_precs = 0.0
    fuzzy_recalls = 0.0
    for yid in gt:
        for g, p in zip(gt[yid], pred[yid]):
            if g:
                total += 1
                # exact
                tp = len(set(g).intersection(set(p)))
                if len(set(p)) > 0:
                    prec = tp / len(set(p))
                else:
                    prec = 0.0
                recall = tp / len(set(g))
                exact_precs += prec
                exact_recalls += recall

                # fuzzy
                fuzzy_tp = 0
                for x in set(g):
                    for y in set(p):
                        if fuzz.ratio(x, y) > 75:
                            fuzzy_tp += 1
                if len(set(p)) > 0:
                    fuzzy_prec = fuzzy_tp / len(set(p))
                else:
                    fuzzy_prec = 0.0
                fuzzy_recall = fuzzy_tp / len(set(g))
                fuzzy_precs += fuzzy_prec
                fuzzy_recalls += fuzzy_recall
    print("Exact Precision: ", exact_precs / total)
    print("Exact Recall: ", exact_recalls / total)
    print("Exact F1: ", 2 * exact_precs * exact_recalls / (total * (exact_precs + exact_recalls)))
    print("Fuzzy Precision: ", fuzzy_precs / total)
    print("Fuzzy Recall: ", fuzzy_recalls / total)
    print("Fuzzy F1: ", 2 * fuzzy_precs * fuzzy_recalls / (total * (fuzzy_precs + fuzzy_recalls)))
    print()


def get_dataset():
    data = defaultdict(list)
    sents = defaultdict(list)
    verbs = defaultdict(list)
    args = defaultdict(list)
    with open("youcook2/reviewed_0812.csv") as gt_f:
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            sent = row['Sentence']
            data[youtube_id].append(int(row["IsUsefulSentence"]))
            sents[youtube_id].append(sent)
            verbs[youtube_id].append(process_gt_verbs(row["Verb"]))
            obj = process_annotation(row["Object(directly related with Verb)"])
            loc = process_annotation(row["Location"])
            time = process_annotation(row["Time"])
            temp = process_annotation(row["Temperature"])
            other = process_annotation(row["Other important phrase(like with"])
            all_args = obj + loc + time + temp + other
            args[youtube_id].append(all_args)
    return data, sents, verbs, args


if __name__ == '__main__':
    vocab = read_vocab('youcook2/1.2.cooking_vocab.strict_filtered.unsorted.lst')

    gt, sents, gt_verbs, gt_args = get_dataset()
    srl_predictor = get_srl_predictor()

    pred_useful = defaultdict(list)
    pred_verbs = defaultdict(list)
    pred_args = defaultdict(list)

    dump_srl_raw_results = {}
    for yid in sents:
        dump_srl_raw_results[yid] = []
        for sent in sents[yid]:
            srl = srl_predictor.predict_json({'sentence': sent})
            srl_chunks = parse_result(srl)
            dump_srl_raw_results[yid].append((sent, srl_chunks))
            filtered_chunks = filter_by_vocab(filter_chunks(srl_chunks), vocab)
            if len(filtered_chunks) == 0:
                pred_args[yid].append([])
                pred_verbs[yid].append([])
                pred_useful[yid].append(0)
            else:
                pred_args[yid].append(get_args(filtered_chunks))
                pred_verbs[yid].append(get_verb(filtered_chunks))
                pred_useful[yid].append(1)

    print("Dumping SRL raw results")
    with open("raw_srl.pkl", 'wb') as raw_file:
        pickle.dump(dump_srl_raw_results, raw_file)

    # sentence isUseful
    y_true_useful = []
    y_pred_useful = []
    for yid in gt:
        assert len(gt[yid]) == len(pred_useful[yid])
        y_true_useful.extend(gt[yid])
        y_pred_useful.extend(pred_useful[yid])

    print(classification_report(y_true_useful, y_pred_useful))

    write_predicted(pred_useful, pred_verbs, pred_args)

    evaluate(gt_verbs, pred_verbs)

    evaluate(gt_args, pred_args)
