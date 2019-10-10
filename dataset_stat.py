import csv
from collections import defaultdict

import nltk
from num2words import num2words
from fuzzywuzzy import fuzz

import spacy

spacy_nlp = spacy.load('en_core_web_sm')

lemmatizer = nltk.stem.WordNetLemmatizer()


def process_annotation(string):
    if string:
        v = []
        for p in string.replace(',', ';').split(";"):
            ps = p.strip().lower()
            if ps:
                v.append(ps)
        return v
    else:
        return []


def transform_numbers(tokens):
    new = []
    for tok in tokens:
        try:
            num = int(tok)
            new.append(num2words(num))
        except:
            new.append(tok)
    return new


def get_toked_ngrams(tokens, n_min=1, n_max=15):
    all_ngrams = []
    for K in range(n_min, n_max + 1):
        for stt in range(0, len(tokens) - K + 1):
            spanning = tokens[stt: stt + K]
            all_ngrams.append({"tokens": spanning, "start": stt, "end": stt + K})

    return all_ngrams


def get_best_match(query, ngrams, threshold=75):
    retrieved = []
    for ngram in ngrams:
        score = fuzz.ratio(query, " ".join(ngram['tokens']))
        if score >= threshold:
            retrieved.append((ngram, score))
    if retrieved:
        return sorted(retrieved, key=lambda x: -x[-1])[0]
    else:
        return None


def get_stat(filter_keys):
    total = 0
    direct_exact_match = 0
    fuzzy_match = 0
    num_words = 0
    key_clip_count = 0
    context_match_count = 0
    ellipsis_count = 0
    coref_count = 0

    # read data
    total_token_num = 0
    total_sent_num = 0
    sentences = defaultdict(list)
    annotations = defaultdict(list)
    with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f:
        reader = csv.DictReader(gt_f)
        for sent_id, row in enumerate(reader):
            youtube_id = row['VideoUrl'].split('?v=')[1]
            sent = row['Sentence'] + ' '  # for later matching
            total_token_num += len(sent.split())
            total_sent_num += 1
            is_useful = int(row['IsUsefulSentence'])
            if is_useful:
                key_clip_count += 1
            verb = process_annotation(row["Verb"])
            obj = process_annotation(row["Object(directly related with Verb)"])
            loc = process_annotation(row["Location"])
            time = process_annotation(row["Time"])
            temp = process_annotation(row["Temperature"])
            other = process_annotation(row["Other important phrase(like with"])
            annotation = {"verb": verb, "obj": obj, "loc": loc, "time": time, "temp": temp, "other": other}
            merged_anns = []
            for k in filter_keys:
                merged_anns.extend(annotation[k])
            sentences[youtube_id].append(sent)
            annotations[youtube_id].append(merged_anns)
    print("Read complete, avg token per sentence: ", total_token_num / total_sent_num)

    for yid in sentences:
        for sent, annotation in zip(sentences[yid], annotations[yid]):
            for a in annotation:
                if not a:
                    continue
                num_words += len(a.split())
                total += 1
                idx = sent.find(a + ' ')
                if idx > 0:
                    direct_exact_match += 1
                else:
                    # transform number phrase
                    doc_a = spacy_nlp(a)
                    query_tokens = [token.text for token in doc_a]
                    new_query = " ".join(transform_numbers(query_tokens))
                    new_idx = sent.find(new_query)
                    if new_idx > 0:
                        direct_exact_match += 1
                    else:
                        # do fuzzy search
                        doc_q = spacy_nlp(sent)
                        tokens = [token.text for token in doc_q]
                        ngrams = get_toked_ngrams(tokens)
                        top1 = get_best_match(new_query, ngrams)
                        if top1:
                            fuzzy_match += 1
                        else:
                            # still not found!? Try to search the context annotations
                            vid_annotations = []
                            for x in annotations[yid]:
                                vid_annotations.extend(x)
                            vid_annotations = set(vid_annotations)
                            for candidate in vid_annotations:
                                if fuzz.ratio(candidate, a) > 80:
                                    context_match_count += 1
                                    if set(tokens).intersection({"it", "its", "they", "them", "their", "theirs"}):
                                        coref_count += 1
                                    else:
                                        ellipsis_count += 1
                                    break


    print("Total number: ", total)
    print("Direct exact match", direct_exact_match / total)
    print("Fuzzy match", fuzzy_match / total)
    print("Total matched percentage", (direct_exact_match + fuzzy_match) / total)
    print("Average #words", num_words/total)
    print("Average #ann per clip", total / key_clip_count)
    print("Context match found", context_match_count/ total)
    print("Coref found", coref_count/ total)
    print("Elipsis found", ellipsis_count/ total)


if __name__ == '__main__':
    get_stat({"verb"})
    get_stat({"obj", "loc", "time", "temp", "other"})
