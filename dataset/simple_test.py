#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sentencepiece as spm

vocab_file = "kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

lines = ["안녕하세요 이제 세해가 되었네요.", "저는 제가 좋아서 코딩을 합니다.", "모두 행복한 새해 되세요!"]

for line in lines:
    pieces = vocab.EncodeAsPieces(line)
    ids = vocab.encode_as_ids(line)
    print(line)  # 단순 문장
    print(pieces)  # 잘라진 부분
    print(ids)  # 그것들의 id
    print("BOS: id", vocab.piece_to_id("으로"))
