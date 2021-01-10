#!/usr/bin/env python
# -*- coding: utf-8 -*-
from MHeadAttention import MultiHeadAttention
from pffn import PoswiseFeedForwardNet
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_head, eps = self.config.layer_norm_epsilon)
        self.pos_ffnn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_head, eps = self.config.layer_norm_epsilon)
        #여기서 inputs의 값은Q라고 생각해도 됨
        #input.shape = [batch_size, n_head, n_seq, d_head]
        #attn_mask.shape = [batch_size, n_seq, n_seq] 
    def forward(self, inputs, attn_mask):
        attn_output, attn_prob = self.self_attn.forward(inputs, inputs, inputs, attn_mask)
        attn_output = self.layer_norm1(inputs+attn_output)
        pos_output = self.pos_ffnn(attn_output)
        output = self.layer_norm2(pos_output+attn_output)
            
        #output.shape = [batch_size, n_seq, d_hidn]
        #attn_prob = [batch_size, n_head, n_seq, n_seq]
        return output, attn_prob



