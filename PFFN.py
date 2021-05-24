#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn

"""
batch_size = bs
dim of token = d_hidn
tokenized input number = n_seq
dim of feedforward network = d_ff
"""


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            in_channels=self.config.d_hidn,
            out_channels=self.config.d_hidn * 4,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.config.d_hidn * 4,
            out_channels=self.config.d_hidn,
            kernel_size=1,
        )
        self.activation = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # bs, d_ff, n_seq
        output = self.activation(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # bs, n_seq, d_hidn
        return output
