import csv
import nltk
from num2words import num2words
from fuzzywuzzy import fuzz

import spacy

spacy_nlp = spacy.load('en_core_web_sm')

lemmatizer = nltk.stem.WordNetLemmatizer()


def process_annotation(string):
    if string:
        v = []
        for p in string.split(";"):
            ps = p.strip().lower()
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


def match_back():
    total = 0
    direct_exact_match = 0
    fuzzy_match = 0
    with open("youcook2/reviewed_0812.csv", newline='', encoding='utf-8') as gt_f, \
            open("youcook2/review.csv", "w", newline='', encoding="utf-8") as out_f:
        reader = csv.DictReader(gt_f)
        fieldnames = ["No", "Title", "VideoUrl", "TimeStamp", "Sentence", "RowNumber", "IsUsefulSentence", "Key steps",
                      "Verb", "Object(directly related with Verb)", "Location", "Time", "Temperature",
                      "Other important phrase(like with", "Verb not found", "Arguments not found", "Number mismatch"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in reader:
            youtube_id = row['VideoUrl'].split('?v=')[1]
            sent = row['Sentence'] + ' '  # for later matching
            verb = process_annotation(row["Verb"])
            obj = process_annotation(row["Object(directly related with Verb)"])
            loc = process_annotation(row["Location"])
            time = process_annotation(row["Time"])
            temp = process_annotation(row["Temperature"])
            other = process_annotation(row["Other important phrase(like with"])
            annotation = {"verb": verb, "obj": obj, "loc": loc, "time": time, "temp": temp, "other": other}
            verb_not_found = 0
            args_not_found = 0
            number_mismatch = 0
            lengths = [len(verb), len(obj), len(loc), len(time), len(temp), len(other)]
            if lengths[0] > 0:
                for length in lengths[1:]:
                    if length > 0 and length != lengths[0]:
                        number_mismatch = 1

            for t in annotation:
                ann = annotation[t]
                if ann:
                    for a in ann:
                        if a:
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
                                        if t == "verb":
                                            verb_not_found = 1
                                        else:
                                            args_not_found = 1

            to_write = dict(row)
            to_write["Verb not found"] = verb_not_found
            to_write["Arguments not found"] = args_not_found
            to_write["Number mismatch"] = number_mismatch
            writer.writerow(to_write)
        print("Direct exact match", direct_exact_match / total)
        print("Fuzzy match", fuzzy_match / total)
        print("Total matched percentage", (direct_exact_match + fuzzy_match) / total)


if __name__ == '__main__':
    match_back()
