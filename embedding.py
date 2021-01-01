#!/usr/bin/env python
# -*- coding: utf-8 -*-
from config import config
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

vocab = config.vocab

lines = ["새해복 많이 받으세요", "저도 많이 받을게요"]


inputs =[]
for line in lines:
    ids = vocab.EncodeAsIds(line)
    ids = torch.tensor(ids)
    inputs.append(ids)


inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

print(inputs.size())
print(inputs)
n_vocab =len(vocab)
d_hidn = 128
nn_emb = nn.Embedding(n_vocab, d_hidn)

input_embs = nn_emb(inputs)
print("input Embedding size : ", input_embs.size())
print("n_vocab : ",n_vocab)


"""
position embedding with sin and cos
"""
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def _cal_angle(position, i_hidn):
        return position / np.power(10000, 2*(i_hidn // 2) / d_hidn)
    def _get_posi_angle_vec(position):
        return [_cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([_get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])

    return sinusoid_table

def cmap():
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('depth')
    plt.xlim((0, d_hidn))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

pos_encoding = get_sinusoid_encoding_table(64, 128)
pos_encoding = torch.FloatTensor(pos_encoding)
nn_pos = nn.Embedding.from_pretrained(pos_encoding, freeze = True) #학습되는 것이 아니니까 freeze 함

position = torch.arange(inputs.size(1), device = inputs.device,dtype=inputs.dtype).expand\
    (inputs.size(0),inputs.size(1))+1

pos_mask = inputs.eq(0)

position.masked_fill_(pos_mask , 0)
pos_embs = nn_pos(position)

print("inputs : ",inputs)
print("positions : ",position)
print("pos_embs_size : ",pos_embs.size())

input_sums = input_embs + pos_embs


"""
Scaled Dot Product Attention 
"""

Q = input_sums
K = input_sums
V = input_sums

print("Q's id : ",id(Q))
print("K's id : ",id(K))
print("V's id : ",id(V))

print("PTR of Q : ",Q.data_ptr())
print("PTR of k : ",K.data_ptr())
print("PTR of v : ",V.data_ptr())

attn_mask = inputs.eq(0)
print("what is attenmask : ",attn_mask)
