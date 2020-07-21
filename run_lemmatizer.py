import spacy
import os
import json
import random
from tqdm import tqdm
import jsonlines


NLP = spacy.load("en_core_sci_lg")

train_file = "data/scicite/train.jsonl"
dev_file = "data/scicite/dev.jsonl"
test_file = "data/scicite/test.jsonl"

tdt = []

for file in [train_file, dev_file, test_file]:
    with open(file, "r") as data_file:
        data = [json.loads(x) for x in list(data_file)]
        tdt.append(data)

train = tdt[0]
dev = tdt[1]
test = tdt[2]

def lemmatize_sentence(sentence):
    doc = NLP(sentence)

    lemmas = []

    for token in doc:
        lemmas.append(token.lemma_)

    return " ".join(lemmas)

for dataset in tdt:
    for i,instance in enumerate(tqdm(dataset)):
        sentence = instance["string"]
        lemmatized = lemmatize_sentence(sentence)
        instance["lemmatized_string"] = lemmatized

with jsonlines.open('data/scicite/train.jsonl', mode='w') as writer:
    writer.write_all(tdt[0])
    
with jsonlines.open('data/scicite/dev.jsonl', mode='w') as writer:
    writer.write_all(tdt[1])
    
with jsonlines.open('data/scicite/test.jsonl', mode='w') as writer:
    writer.write_all(tdt[2])
