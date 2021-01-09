#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sdpAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.module):
    def __init__(self, d_hidn, n_head, d_head): #dim of word, num of head, dim of head
        super().init()
        self.d_hidn = d_hidn
        self.n_head = n_head
        self.d_head = d_head


        self.W_Q = nn.Linear(d_hidn, n_head * d_head)
        self.W_K = nn.Linear(d_hidn, n_head * d_head)
        self.W_v = nn.Linear(d_hidn, n_head * d_head)
        self.scaled_dot_attn = sdpAttention(d_head)
        self.linear = nn.Linear(n_head * d_head , d_hidn)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_s = W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) #for one head you have number of input which consist of d_head
        k_s = W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        v_s = W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        #context = batch_size, n_head, num_of_word, d_head
        #attn_prob = batch_size, n_head, num_of_token, num_of_token
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, attn_mask)

        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_head*self.d_head)

        output = self.linear(context)

        return output,attn_prob
