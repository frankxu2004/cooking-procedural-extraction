import json

if __name__ == '__main__':
    with open('youcook2/2.cooking_vocab_filtered_captions.tmp.json', encoding='utf-8') as f, \
            open('youcook2/all_captions.txt', 'w', encoding='utf-8') as out_f, \
            open('youcook2/all_youtube_ids.txt', 'w', encoding='utf-8') as ytb_f:
        data = json.load(f)
        for item in data:
            youtube_id = item['youtube_id']
            ytb_f.write(youtube_id + '\n')
            # write caption texts
            with open("youcook2/split/" + youtube_id + '.txt', 'w', encoding='utf-8') as of:
                for caption in item["caption_transcript_pair_list"]:
                    out_f.write(caption['sentence'] + '\n')
                    of.write(caption['sentence'] + '\n')
