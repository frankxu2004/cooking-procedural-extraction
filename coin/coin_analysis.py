import json

with open('COIN.json', encoding='utf-8') as json_file:
    jsonobj = json.load(json_file)

steps = set()


for vid in jsonobj['database']:
    for step in jsonobj['database'][vid]['annotation']:
        steps.add(step['label'])

print(steps)
print(len(steps))