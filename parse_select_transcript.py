import csv
import nltk
from collections import defaultdict

from sklearn.metrics import classification_report

from utils import *

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
        for p in verb_str.split(";"):
            ps = p.strip().lower()
            if ps:
                # get root verb only
                v.append(ps.split()[0])
        return v
    else:
        return []


def get_dataset():
    data = defaultdict(list)
    sents = defaultdict(list)
    verbs = defaultdict(list)
    with open("youcook2/Video_data_all.csv") as gt_f:
        reader = csv.DictReader(gt_f)
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            sent = row['Sentence']
            try:
                data[youtube_id].append(int(row["IsUsefulSentence"]))
                sents[youtube_id].append(sent)
                verbs[youtube_id].append(process_gt_verbs(row["Verb"]))
            except ValueError:
                continue
    return data, sents, verbs


def get_youtube_id_from_file(file):
    return file.split('/')[-1].split('.txt')[0]


def get_youtube_ids_from_files(files):
    youtube_ids = []
    for file in files:
        youtube_ids.append(get_youtube_id_from_file(file))
    return youtube_ids


if __name__ == '__main__':
    vocab = read_vocab('youcook2/1.2.cooking_vocab.strict_filtered.unsorted.lst')
    src_dir = "youcook2/select"
    target_dir = src_dir + '-sent'

    gt, sents, gt_verbs = get_dataset()

    srl_predictor = get_srl_predictor()

    pred = defaultdict(list)
    pred_verbs = defaultdict(list)

    for yid in sents:
        for sent in sents[yid]:
            srl = srl_predictor.predict_json({'sentence': sent})
            srl_chunks = parse_result(srl)

            filtered_chunks = filter_by_vocab(filter_chunks(srl_chunks), vocab)
            if len(filtered_chunks) == 0:
                pred_verbs[yid].append([])
                pred[yid].append(0)
            else:
                pred_verbs[yid].append(get_verb(filtered_chunks))
                pred[yid].append(1)

    y_true = []
    y_pred = []
    for yid in gt:
        print(yid)
        assert len(gt[yid]) == len(pred[yid])
        y_true.extend(gt[yid])
        y_pred.extend(pred[yid])

    print(classification_report(y_true, y_pred))

    with open("verb_eval.txt", 'w', encoding='utf-8') as of:
        # calculate verb scores
        em_num = 0
        total = 0
        hamming = 0.0
        for yid in gt_verbs:
            print(yid)
            of.write(yid + '\n')
            for g, p, sent in zip(gt_verbs[yid], pred_verbs[yid], sents[yid]):
                if g:
                    of.write(sent + '\n' + " ".join(g) + "\n" + " ".join(p) + '\n\n')
                    total += 1
                    if set(g) == set(p):
                        em_num += 1
                    hamming += len(set(g).intersection(set(p))) / len(set(g).union(set(p)))

    print("EM Acc: ", em_num / total)
    print("Hamming: ", hamming / total)
