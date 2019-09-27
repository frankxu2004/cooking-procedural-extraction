import json
import csv

from tqdm import tqdm
from utils import *


if __name__ == '__main__':

    oie_predictor = get_oie_predictor()
    srl_predictor = get_srl_predictor()

    with open('youcook2/test_samples.csv', encoding='utf-8') as matched_file, \
            open('youcook2/test_samples_srl_on_matched.csv', 'w', newline='', encoding='utf-8') as srl_outfile, \
            open('youcook2/test_samples_oie_on_matched.csv', 'w', newline='', encoding='utf-8') as oie_outfile:
        csv_reader = csv.reader(matched_file)
        srl_csvwriter = csv.writer(srl_outfile)
        oie_csvwriter = csv.writer(oie_outfile)
        for item in tqdm(csv_reader):
            sent = item[3]

            oie = oie_predictor.predict_json({'sentence': sent})
            oie_chunks = parse_result(oie)

            srl = srl_predictor.predict_json({'sentence': sent})
            srl_chunks = parse_result(srl)

            srl_csvwriter.writerow(item + all_chunks_to_string(srl_chunks))
            oie_csvwriter.writerow(item + all_chunks_to_string(oie_chunks))

