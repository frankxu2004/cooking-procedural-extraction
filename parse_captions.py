import csv

import nltk

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

if __name__ == '__main__':
    vocab = read_vocab('youcook2/1.2.cooking_vocab.strict_filtered.unsorted.lst')
    srl_predictor = get_srl_predictor()

    with open('youcook2/urlsegmentcaption.txt', encoding='utf-8') as infile, \
        open('youcook2/urlsegmentcaption_srl_parsed.txt', 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile)
        for row in reader:
            sent = row[3]
            srl = srl_predictor.predict_json({'sentence': sent})
            srl_chunks = parse_result(srl)
            filtered_chunks = filter_by_vocab(filter_chunks(srl_chunks), vocab)
            srl_v = get_verb(srl_chunks)
            srl_a = get_args(srl_chunks)
            filtered_v = get_verb(filtered_chunks)
            filtered_a = get_args(filtered_chunks)
            to_write = list(row)
            to_write.extend([srl_chunks, srl_v, srl_a, filtered_chunks, filtered_v, filtered_a])
            writer.writerow(to_write)

