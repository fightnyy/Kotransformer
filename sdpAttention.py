#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = 1/(d_head**0.5)

    def forward (self, Q, K, V, attn_mask):
        score = torch.matmul(Q, K.transpose(-2, -1)).mul(self.scale)
        score.mask_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim = -1)(score)
        context = torch.matmul(attn_prob, V)
        return context , attn_prob 
