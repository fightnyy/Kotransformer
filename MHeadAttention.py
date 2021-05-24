#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sdpAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import pdb


class MultiHeadAttention(nn.Module):
    def __init__(self, config):  # dim of word, num of head, dim of head
        super().__init__()
        self.config = config
        self.d_hidn = self.config.d_hidn
        self.n_head = self.config.n_head
        self.d_head = self.config.d_head
        self.dropout = nn.Dropout(config.dropout)

        self.W_Q = nn.Linear(self.d_hidn, self.n_head * self.d_head)
        self.W_K = nn.Linear(self.d_hidn, self.n_head * self.d_head)
        self.W_V = nn.Linear(self.d_hidn, self.n_head * self.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.linear = nn.Linear(self.n_head * self.d_head, self.d_hidn)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_s = (
            self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        )  # for one head you have number of input which consist of d_head
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # context = batch_size, n_head, num_of_word, d_head
        # attn_prob = batch_size, n_head, num_of_token, num_of_token
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_head * self.d_head)
        )

        output = self.linear(context)
        output = self.dropout(output)
        return output, attn_prob
