import csv
import pickle
from collections import defaultdict

import json
import nltk
from munkres import Munkres, make_cost_matrix
from fuzzywuzzy import fuzz
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils import get_srl_predictor, read_vocab, parse_result, filter_chunks, get_args, get_verb

lemmatizer = nltk.stem.WordNetLemmatizer()

def filter_by_vocab(chunks, vocab):
    ret = []
    for chunk in chunks:
        for idx, c in enumerate(chunk):
            if c['type'] == "V":
                lemma = lemmatizer.lemmatize(c['text'], 'v')
                if lemma in vocab:
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
                # v.append(ps.split()[0])
                v.append(ps)
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


def write_pred_summarize(pred_summarize):
    with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f, \
            open("youcook2/reviewed_0812_pred_for_summarize.csv", "w", newline='', encoding="utf-8") as out_f:
        fieldnames = ["No", "Title", "VideoUrl", "TimeStamp", "Sentence", "RowNumber", "IsUsefulSentence", "Key steps",
                      "Verb", "Object(directly related with Verb)", "Location", "Time", "Temperature",
                      "Other important phrase(like with", "PredForSummarize"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            row_num = int(row['RowNumber'])
            to_write = dict(row)
            to_write['PredForSummarize'] = pred_summarize[youtube_id][row_num]
            writer.writerow(to_write)

def evaluate_keysent(gt, pred):
    y_true_useful = []
    y_pred_useful = []
    for yid in gt:
        assert len(gt[yid]) == len(pred[yid])
        y_true_useful.extend(gt[yid])
        y_pred_useful.extend(pred[yid])
    print("Precision: ", "{:.1f}".format(precision_score(y_true_useful, y_pred_useful, average='binary') * 100))
    print("Recall: ", "{:.1f}".format(recall_score(y_true_useful, y_pred_useful, average='binary') * 100))
    print("F1 Score: ", "{:.1f}".format(f1_score(y_true_useful, y_pred_useful, average='binary') * 100 ))
    print("Acc: ", "{:.1f}".format(accuracy_score(y_true_useful, y_pred_useful) * 100))
    print()

def gt_compare(g, sota_pred, yid, idx):
    if sota_pred and sota_pred[yid][idx]:
        return True
    elif g:
        return True
    else:
        return False



def evaluate(gt, pred, fuzzy=False, partial_ratio=True, sota_pred=None):
    m = Munkres()
    total = 0
    precs = 0.0
    recalls = 0.0
    for yid in gt:
        for i, (g, p) in enumerate(zip(gt[yid], pred[yid])):
            if gt_compare(g, sota_pred, yid, i):
                total += 1
                if not fuzzy:
                    # exact
                    tp = len(set(g).intersection(set(p)))
                    if len(set(p)) > 0:
                        prec = tp / len(set(p))
                    else:
                        prec = 0.0
                    if len(set(g)) > 0:
                        recall = tp / len(set(g))
                    else:
                        recall = 0.0
                    precs += prec
                    recalls += recall
                else:
                    # fuzzy
                    fuzzy_tp = 0.0
                    if len(set(p)) > 0 and len(set(g)) > 0:
                        profit_matrix = []
                        for idx, x in enumerate(set(g)):
                            profit_matrix.append([])
                            for y in set(p):
                                if partial_ratio:
                                    score = fuzz.partial_ratio(x, y)
                                else:
                                    score = fuzz.ratio(x, y)
                                profit_matrix[idx].append(score)
                                # if score > 75:
                                #     fuzzy_tp += 1
                                #     break
                        indexes = m.compute(make_cost_matrix(profit_matrix))
                        for row, column in indexes:
                            value = profit_matrix[row][column]
                            fuzzy_tp += value
                        fuzzy_tp /= 100

                    if len(set(p)) > 0:
                        fuzzy_prec = fuzzy_tp / len(set(p))
                    else:
                        fuzzy_prec = 0.0
                    if len(set(g)) > 0:
                        fuzzy_recall = fuzzy_tp / len(set(g))
                    else:
                        fuzzy_recall = 0.0
                    precs += fuzzy_prec
                    recalls += fuzzy_recall
    if not fuzzy:
        print("Exact:")
    elif partial_ratio:
        print("Partial Fuzzy:")
    else:
        print("Fuzzy:")
    if sota_pred:
        print("Use predicted")
    else:
        print("Use ground truth")
    print("Precision: ", "{:.1f}".format(precs / total * 100))
    print("Recall: ", "{:.1f}".format(recalls / total * 100))
    print("F1: ", "{:.1f}".format(2 * precs * recalls / (total * (precs + recalls)) * 100))
    print()


def get_pred_key_sent():
    pred_keysent = defaultdict(list)
    with open("youcook2/reviewed_0812.attach_key_sent_selection.tsv") as gt_f:
        reader = csv.DictReader(gt_f, delimiter='\t')
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            pred_keysent[youtube_id].append(int(row["key_sentence_prediction"]))
    return pred_keysent


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

def get_ordered_pred_for_summarize(chunks):
    ordered_tups = []
    for idx, tup in enumerate(chunks):
        for item in tup:
            if item['type'] == 'V':
                ordered_tups.append([('V', item['text'])])
                break
        for item in sorted(tup, key = lambda i: i['type']):
            if item['type'] != 'V':
                ordered_tups[idx].append((item['type'], item['text']))
    return ordered_tups

def srl_post_heuristics(srl_chunks, yid, vocab, pred_useful, pred_verbs, pred_args):
    filtered_chunks = filter_by_vocab(filter_chunks(srl_chunks), vocab)
    if len(filtered_chunks) == 0:
        pred_args[yid].append([])
        pred_verbs[yid].append([])
        pred_useful[yid].append(0)
    else:
        pred_args[yid].append(get_args(filtered_chunks))
        pred_verbs[yid].append(get_verb(filtered_chunks))
        pred_useful[yid].append(1)
    return filtered_chunks


if __name__ == '__main__':
    vocab = read_vocab('youcook2/1.2.cooking_vocab.strict_filtered.unsorted.lst')
    use_existing_srl_results = True

    gt, sents, gt_verbs, gt_args = get_dataset()

    sota_pred_keysent = get_pred_key_sent()

    pred_useful = defaultdict(list)
    pred_verbs = defaultdict(list)
    pred_args = defaultdict(list)

    pred_for_summarize = defaultdict(list)

    if use_existing_srl_results:
        print("Reading SRL raw results")
        dump_srl_raw_results = pickle.load(open("raw_srl.pkl", 'rb'))
        for yid in sents:
            assert len(sents[yid]) == len(dump_srl_raw_results[yid])
            for idx, sent in enumerate(sents[yid]):
                srl_chunks = dump_srl_raw_results[yid][idx][1]
                filtered_chunks = srl_post_heuristics(srl_chunks, yid, vocab, pred_useful, pred_verbs, pred_args)
                pred_for_summarize[yid].append(get_ordered_pred_for_summarize(filtered_chunks))

    else:
        srl_predictor = get_srl_predictor()
        dump_srl_raw_results = {}
        for yid in sents:
            dump_srl_raw_results[yid] = []
            for sent in sents[yid]:
                srl = srl_predictor.predict_json({'sentence': sent})
                srl_chunks = parse_result(srl)
                dump_srl_raw_results[yid].append((sent, srl_chunks))
                srl_post_heuristics(srl_chunks, yid, vocab, pred_useful, pred_verbs, pred_args)

        print("Dumping SRL raw results")
        with open("raw_srl.pkl", 'wb') as raw_file:
            pickle.dump(dump_srl_raw_results, raw_file)


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

    write_predicted(pred_useful, pred_verbs, pred_args)
    write_pred_summarize(pred_for_summarize)
    json.dump([pred_verbs, pred_args], open('youcook2/srl_pred.json', 'w', encoding='utf-8'))