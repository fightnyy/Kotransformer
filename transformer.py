#!/usr/bin/env python
# -*- coding: utf-8 -*-

from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attn_probs = self.encoder.forward(enc_inputs)

        dec_outputs, self_attn_probs, dec_enc_attn_probs = self.decoder(
            dec_inputs, enc_inputs, enc_outputs
        )

        return dec_outputs, enc_self_attn_probs, self_attn_probs, dec_enc_attn_probs
