#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from config import config


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.config = config.tconfig
        self.dropout = nn.Dropout(self.config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        score = torch.matmul(Q, K.transpose(-1, -2)).mul(self.scale)
        score.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(score)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)
        return context, attn_prob
