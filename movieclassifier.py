#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn
from transformer import Transformer

class MovieClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias = False)

    
    def forward(self, enc_inputs, dec_intputs)
        
        """
        # dec_outputs.size = [batch_size, n_dec_seq_len, d_hidn]
        # enc_self_attn_probs = [batch_size, n_enc_q_len, n_enc_k_len]
        # self_attn_probs = [batch_size, n_dec_q_len, n_dec_k_len]
        # dec_enc_attn_probs = [batch_size, n_enc_q_len, n_dec_k_len]
        """
        dec_outputs, enc_self_attn_probs, self_attn_probs, dec_enc_attn_probs = self.transformer.forward(enc_inputs, dec_inputs)
        
        """
        dec_outputs 가 나타내려고 하는것
        
        # 어디에 각 token들의 임베딩 값 중 가장 큰것이 무엇이나?
        # dec_outputs.size =[batch_size, d_hidn]
        """
        dec_outputs,_ =torch.max(dec_outputs, dim = 1)
        
        #logits.shape = [batch_size, n_output]
        logits = self.projection(dec_outputs)

        """
        # logtis.size = [batch_size, n_outputs]
        # enc_self_attn_probs = [batch_size, n_head, n_enc_seq_len, n_enc_seq_len]
        # self_attn_probs = [batch_size, n_head, n_dec_seq_len, n_dec_seq_len]
        # dec_enc_attn_probs = [batch_Size, n_head_ n_enc_seq_len, n_dec_seq_len]
        """
        return logits, enc_self_attn_probs, self_attn_probs, dec_enc_attn_probs
