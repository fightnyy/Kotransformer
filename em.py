#!/usr/bin/env python
# -*- coding: utf-8 -*-
from config import config
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

vocab = config.vocab

lines = ["새해복 많이 받으세요", "저도 많이 받을게요"]


inputs = []
for line in lines:
    ids = vocab.EncodeAsIds(line)
    ids = torch.tensor(ids)
    inputs.append(ids)


inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

print(inputs.size())
print(inputs)
n_vocab = len(vocab)
d_hidn = 128
nn_emb = nn.Embedding(n_vocab, d_hidn)

input_embs = nn_emb(inputs)
print("input Embedding size : ", input_embs.size())
print("n_vocab : ", n_vocab)


"""
position embedding with sin and cos
"""


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def _cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

    def _get_posi_angle_vec(position):
        return [_cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([_get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return sinusoid_table


def cmap():
    plt.pcolormesh(pos_encoding, cmap="RdBu")
    plt.xlabel("depth")
    plt.xlim((0, d_hidn))
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()


pos_encoding = get_sinusoid_encoding_table(64, 128)
pos_encoding = torch.FloatTensor(pos_encoding)
nn_pos = nn.Embedding.from_pretrained(
    pos_encoding, freeze=True
)  # 학습되는 것이 아니니까 freeze 함

position = (
    torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(
        inputs.size(0), inputs.size(1)
    )
    + 1
)

pos_mask = inputs.eq(0)

position.masked_fill_(pos_mask, 0)
pos_embs = nn_pos(position)

print("inputs : ", inputs)
print("positions : ", position)
print("pos_embs_size : ", pos_embs.size())

input_sums = input_embs + pos_embs


"""
Scaled Dot Product Attention 
"""

Q = input_sums  # size (2, 8 , 128)
K = input_sums
V = input_sums

print("Q's id : ", id(Q))
print("K's id : ", id(K))
print("V's id : ", id(V))

print("PTR of Q : ", Q.data_ptr())
print("PTR of k : ", K.data_ptr())
print("PTR of v : ", V.data_ptr())

attn_mask = inputs.eq(0).unsqueeze(1).expand(Q.size(0), Q.size(1), K.size(1))
scores = torch.matmul(Q, K.transpose(-1, -2))
d_head = 64
scores.mul_(1 / d_head ** 0.5)
scores = scores.masked_fill(attn_mask, -1e9)
attn_prob = nn.Softmax(dim=-1)(scores)  # 가장 연관성이 높은 값
attn_prob = torch.tensor(attn_prob)
context = torch.matmul(attn_prob, V)

n_head = 2


W_Q = nn.Linear(d_hidn, n_head * d_head)
W_K = nn.Linear(d_hidn, n_head * d_head)
W_V = nn.Linear(d_hidn, n_head * d_head)
print("W_Q 가 그래서 뭔데?", W_Q)
batch_size = Q.size(0)
q_s = W_Q(Q)  # 곱하는 결과를 발생함(정확히는 matmul)
k_s = W_K(K)
v_s = W_V(V)
batch_size = 2
q_s = q_s.view(batch_size, -1, n_head, d_head).transpose(1, 2)
k_s = k_s.view(batch_size, -1, n_head, d_head).transpose(1, 2)
v_s = v_s.view(batch_size, -1, n_head, d_head).transpose(1, 2)


print("attn_mask : ", attn_mask)
