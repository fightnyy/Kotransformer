#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sentencepiece as spm
import pandas as pd
import json

vocab_file = "kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

"""
vocab을 tokenize 해서 json 파일 형태로 저장
미리 tokenize 하지 않으면 training시 매번
다시 해줘야 한다. => 시간 오래걸림.
"""

def prepare_train(vocab, infile, outfile):
    df = pd.read_csv(infile, sep = '\t', engine='python')
    with open(outfile, "w")as f:
        for index, row in df.iterrows():
            document = row['document']
            if type(document) != str:
                continue
            instance = { "id": row['id'], "doc": vocab.encode_as_pieces(document), "label": row['label']}
            f.write(json.dumps(instance))
            f.write("\n")


def main():
    prepare_train(vocab, "ratings_train.txt", "ratings_train.json")
    prepare_train(vocab,"ratings_test.txt", "ratings_test.json" )
if __name__ == '__main__':
    main()
    
