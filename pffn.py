#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn.functional as F

"""
batch_size = bs
dim of token = d_hidn
tokenized input number = n_seq
dim of feedforward network = d_ff
"""

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_hidn):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels = d_hidn, out_channels = d_hidn * 4 , kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels = d_hidn*4 , out_channles = d_hidn, kernel_size = 1)
        self.activation = F.gelu


    def forward(self, inputs):
        #bs, d_ff, n_seq
        output = self.activation(self.conv1(inputs.transpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        
        #bs, n_seq, d_hidn
        return output

