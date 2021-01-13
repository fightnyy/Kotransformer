#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from encoderlayer import EncoderLayer

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def _cal_angle(position, i_hidn):
        return position / np.power(10000, 2*(i_hidn // 2) / d_hidn)
    def _get_posi_angle_vec(position):
        return [_cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]
     
    #각 position별 hidden index별 angle값을 구한다
    sinusoid_table = np.array([_get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    #Embedding 값중 짝수에 해당하는 곳에만 sin 값을 취함
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    #Embedding 값 중 홀수에 해당하는 곳에만 cos 값을 취함
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])

    return sinusoid_table

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)

    return pad_attn_mask






class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze = True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

        #inputs.size = [batch_size, n_seq_len]
    def forward(self, inputs):
        position = torch.arange(inputs.size(1), device = inputs.device , dtype = inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous()+1
        pos_mask = inputs.eq(self.config.i_pad)
        position.masked_fill_(pos_mask, 0)
            
        #outputs.size = [batch_size, n_seq_len(token들의 길이), d_hidn]
        outputs = self.enc_emb(inputs) + self.pos_emb(position)


        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)


        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer.forward(outputs, attn_mask)
            attn_probs.append(attn_prob)
            
        #output.size = [batch_size, n_seq_len, d_hidn]
        #attn_probs [n_layer, batch_size, n_seq_len, n_seq_len]
        return outputs, attn_probs



