import os

import torch

from allennlp.models import load_archive
from allennlp.predictors.predictor import Predictor


def read_vocab(vocab_file):
    vocab = set()
    with open(vocab_file, encoding='utf-8') as vf:
        for line in vf:
            vocab.add(line.strip())
    return vocab


def get_txt_files(dir_path):
    file_list = []
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            file_list.append(os.path.join(dir_path, file))
    return file_list


def get_oie_predictor():
    if torch.cuda.is_available():
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz",
                               cuda_device=0)
    else:
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
    return Predictor.from_archive(archive)


def get_srl_predictor():
    if torch.cuda.is_available():
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                               cuda_device=0)
    else:
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")

    return Predictor.from_archive(archive)


def bio_to_frames(tags, tokens):
    frame = []
    chunk = []
    assert len(tags) == len(tokens)
    chunk_type = ''
    for (token, tag) in zip(tokens, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                frame.append({'text': " ".join(chunk), 'type': chunk_type})
                chunk = []

            if tag.startswith("B-"):
                chunk_type = tag[2:]
                chunk.append(token)

    if chunk:
        frame.append({'text': " ".join(chunk), 'type': chunk_type})

    return frame


def parse_result(oie_results):
    verbs = oie_results['verbs']
    words = oie_results['words']
    all_chunks = []
    for v in verbs:
        chunk = bio_to_frames(v['tags'], words)
        all_chunks.append(chunk)
    return all_chunks


excluded_types = {"ARGM-CAU", "ARGM-DIS", "ARGM-MOD", "ARGM-NEG", "ARGM-EXT", "ARGM-PNC", "ARGM-REC"}

def filter_arguments(chunks):
    new_chunks = []
    for c in chunks:
        if c['type'] != "ARG0" and "ARG" in c['type'] and c['type'] not in excluded_types:
            new_chunks.append(c)
        elif c['type'] == 'V':
            new_chunks.append(c)
    if len(new_chunks) <= 1:
        return None
    else:
        return new_chunks


def chunks_to_string(chunks):
    ret = ''
    for c in chunks:
        ret += c['text'] + '|' + c['type'] + '|'
    return ret


def all_chunks_to_string(all_chunks):
    ret = []
    for x in all_chunks:
        filtered = filter_arguments(x)
        if filtered:
            ret.append(chunks_to_string(filtered))
    return ret


def filter_chunks(all_chunks):
    ret = []
    for x in all_chunks:
        filtered = filter_arguments(x)
        if filtered:
            ret.append(filtered)
    return ret


def get_verb(chunks):
    ret = []
    for x in chunks:
        for arg in x:
            if arg['type'] == 'V':
                ret.append(arg['text'])
    return ret


def get_args(chunks):
    ret = []
    for x in chunks:
        for arg in x:
            if arg['type'].startswith('ARG'):
                ret.append(arg['text'])
    return ret


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def get_youtube_id_from_file(file):
    return file.split('/')[-1].split('.txt')[0]


def get_youtube_ids_from_files(files):
    youtube_ids = []
    for file in files:
        youtube_ids.append(get_youtube_id_from_file(file))
    return youtube_ids


