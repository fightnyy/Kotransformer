#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from MHeadAttention import MultiHeadAttention
from PFFN import PoswiseFeedForwardNet

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.self_attn = MultiHeadAttention(self.config)
        self.self_layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.self_layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.self_layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)

    # 디코더 사진을 보면 알 수 있듯이 맨처음엔 디코더끼리 셀프어텐션을 취하고
    # 그 이후 인코더의 Key, Query와 디코더 셀프어텐션 한 Value끼리 어텐션을 취하기 때문에
    # 인코더에 비해 입력파라미터 수가  많은 것을 볼 수 있다
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):

        self_att_outputs, self_attn_prob = self.self_attn.forward(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.self_layer_norm1(self_att_outputs+dec_inputs)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn.forward(self_att_outputs, self_att_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_att_outputs = self.self_layer_norm2(self.dec_enc_att_outputs + self_att_outputs)

        ffn_outputs = self.pos_ffn.forward(dec_enc_att_outputs)
        ffn_outputs = self.self_layer_norm3(ffn_outputs)
        
        #ffn_outputs.size = [batch_size, n_dec_seq_len, d_hidn]
        #self_attn_prob.size = [batch_size, n_head,n_dec_seq_len, n_dec_seq_len]
        #dec_enc_attn_prob = [batch_size, n_head,n_dec_seq_len, n_enc_seq_len]
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob
    

        
