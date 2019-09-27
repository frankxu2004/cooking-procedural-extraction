import json

if __name__ == '__main__':
    with open('youcook2/3.video_item_list.with_transcript.azure_stt.youtube_auto_generated_filtered.json',
              encoding='utf-8') as f:
        dataset = json.load(f)
        print(len(dataset))

    youtube_ids = set()
    # with open('youcook2/all_youtube_ids.txt', encoding='utf-8') as f:
    #     for line in f:
    #         youtube_ids.add(line.strip())

    with open('youcook2/select_youtube_urls.txt', encoding='utf-8') as f:
        for line in f:
            youtube_ids.add(line.strip().split('?v=')[1])

    with open('youcook2/select_transcripts.txt', 'w', encoding='utf-8') as out_f:
        for item in dataset:
            youtube_id = item['youtube_id']
            if youtube_id in youtube_ids:
                # write transcript texts
                with open("youcook2/select/" + youtube_id + '.txt', 'w', encoding='utf-8') as of:
                    for trans in item['transcript']['en']['azure_stt_transcripts']['sentence_level']:
                        out_f.write(trans['text'] + '\n')
                        of.write(trans['text'] + '\n')
