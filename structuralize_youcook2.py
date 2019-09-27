import json
import csv

from tqdm import tqdm
from utils import *


if __name__ == '__main__':
    print("Reading data file...")

    oie_predictor = get_oie_predictor()
    srl_predictor = get_srl_predictor()

    with open('youcook2/2.cooking_vocab_filtered_captions.tmp.json.matched.json', encoding='utf-8') as matched_file:
        matched_data = json.load(matched_file)

    captions_sents_batch = []
    src_sents_batch = []
    for item in matched_data:
        transcript_sen_ids = item['transcript_sen_ids']
        transcript_tokens = item['transcript_tokens']
        assert len(transcript_sen_ids) == len(transcript_tokens)
        id2sents = dict(zip(transcript_sen_ids, transcript_tokens))
        src_matched = item['src_matched']
        if src_matched:
            matched_sent_id = src_matched['src_sens_id']
            src_matched_sent = ' '.join(id2sents[matched_sent_id])
        else:
            src_matched_sent = 'NULL'
        caption = ' '.join(item['caption_tokens'])

        captions_sents_batch.append({'sentence': caption})
        src_sents_batch.append({'sentence': src_matched_sent})

    print("Begin batch predict")
    # Batch predict
    caption_oie_all = []
    src_oie_all = []
    caption_srl_all = []
    src_srl_all = []
    for b in tqdm(chunks(captions_sents_batch, 128)):
        caption_oie_all += oie_predictor.predict_batch_json(b)
    for b in tqdm(chunks(src_sents_batch, 128)):
        src_oie_all += oie_predictor.predict_batch_json(b)

    for b in tqdm(chunks(captions_sents_batch, 128)):
        caption_srl_all += srl_predictor.predict_batch_json(b)
    for b in tqdm(chunks(src_sents_batch, 128)):
        src_srl_all += srl_predictor.predict_batch_json(b)

    print("Writing results...")
    with open('youcook2/srl_on_matched.csv', 'w', newline='', encoding='utf-8') as srl_outfile, \
            open('youcook2/oie_on_matched.csv', 'w', newline='', encoding='utf-8') as oie_outfile:
        srl_csvwriter = csv.writer(srl_outfile)
        oie_csvwriter = csv.writer(oie_outfile)
        for idx, item in enumerate(matched_data):
            vid_id = item['vid_id']
            caption_id = item['caption_id']

            caption = captions_sents_batch[idx]['sentence']
            src_matched_sent = src_sents_batch[idx]['sentence']

            caption_oie = caption_oie_all[idx]
            src_oie = src_oie_all[idx]
            caption_oie_chunks = parse_result(caption_oie)
            src_oie_chunks = parse_result(src_oie)

            caption_srl = caption_srl_all[idx]
            src_srl = src_srl_all[idx]
            caption_srl_chunks = parse_result(caption_srl)
            src_srl_chunks = parse_result(src_srl)

            srl_csvwriter.writerow([vid_id, caption_id, caption, src_matched_sent] +
                                   all_chunks_to_string(caption_srl_chunks) + ['|||'] +
                                   all_chunks_to_string(src_srl_chunks))
            oie_csvwriter.writerow([vid_id, caption_id, caption, src_matched_sent] +
                                   all_chunks_to_string(caption_oie_chunks) + ['|||'] +
                                   all_chunks_to_string(src_oie_chunks))
