#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from encoder import get_sinusoid_encoding_table
from encoder import get_attn_pad_mask
import torch.nn as nn
from decoder_layer import DecoderLayer

def get_attn_decoder_mask(seq):
    batch_size, len_seq = seq.size(0), seq.size(1)
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(batch_size, len_seq, len_seq)
    subsequent_mask = torch.triu(diagonal=1)# 대각선 기준으로  위에가 0인상태대각은 모두 1

    return subsequent_mask



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq+1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze = True)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in self.config.n_layer])

    #dec_inputs.size =[batch_size, n_dec_seq_len]
    def forward(self, dec_inputs, enc_inputs, enc_outputs):

        position = torch.arange(dec_inputs.size(1), dtype =dec_inputs.dtype, device = dec_inputs.device).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(0)
        position.masked_fill_(pos_mask, 0)

        #dec_outputs.size = [batch_size, n_dec_seq_len, d_hidn]
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(position)

        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, config.i_pad)

        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, config.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []

        for layer in self.layers:

            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer.forward(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)

        return dec_outputs, self_attn_probs, dec_enc_attn_probs

