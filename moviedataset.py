#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import json

from tqdm import tqdm
from config import config


"""데이터 로더"""
def build_data_loader(vocab, infile, args, shuffle = True):
    dataset = MovieDataset(vocab, infile)
    if 1< args.n_gpu and shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.tconfig.batch_size, sampler = sampler, collate_fn =movie_collate_fn)
    else :
        sampler = None
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.tconfig.batch_size, sampler=sampler, shuffle=shuffle, collate_fn = movie_collate_fn) 
    return loader, sampler

def movie_collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    #encoder 와 decoder의 길이가 같아지도록 만들기 위해서 pad를 붙임
    enc_inputs = nn.utils.rnn.pad_sequence(enc_inputs, batch_first = True, padding_value = 0)
    dec_inputs = nn.utils.rnn.pad_sequence(dec_inputs, batch_first = True, padding_value = 0)

    #Label은  길이가 1 고정이어서 stack 함수를 이용하여 tensor로 만듬
    batch = [torch.stack(labels, dim =0), enc_inputs, dec_inputs]
    return batch

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentence = []

        line_cnt = 0
        with open(infile, 'r')as f:
            for line in f :
                line_cnt +=1 

        with open(infile, 'r')as f:
            for i, line in enumerate(tqdm(f, desc =f"Loading {infile}",total = line_cnt, unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                self.sentence.append([vocab.piece_to_id(p) for p in data["doc"]])

    def __len__(self):
        assert len(self.labels) == len(self.sentence)
        return len(self.labels)


    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        sentence = torch.tensor(self.sentence[item])
        BOS_id = torch.tensor([self.vocab.piece_to_id("[BOS]")])
        return (label, sentence, BOS_id)


