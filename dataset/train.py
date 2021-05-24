#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sentencepiece as spm

corpus = "kowiki.txt"
prefix = "kowiki"
vocab_size = 8000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size+7}"
    + " --model_type=bpe"
    + " --max_sentence_length=999999"
    + " --pad_id=0 --pad_piece=[PAD]"
    + " --unk_id=1 --unk_piece=[UNK]"
    + " --bos_id=2 --bos_piece=[BOS]"
    + " --eos_id=3 --eos_piece=[EOS]"
    + " --user_defined_symbols=[SEP],[CLS],[MASK]"
)
