#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def _cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

    def _get_posi_angle_vec(position):
        return [_cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    # 각 position별 hidden index별 angle값을 구한다
    sinusoid_table = np.array([_get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    # Embedding 값중 짝수에 해당하는 곳에만 sin 값을 취함
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # Embedding 값 중 홀수에 해당하는 곳에만 cos 값을 취함
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return sinusoid_table


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)

    return pad_attn_mask


def get_attn_decoder_mask(seq):
    batch_size, len_seq = seq.size(0), seq.size(1)
    subsequent_mask = (
        torch.ones_like(seq).unsqueeze(-1).expand(batch_size, len_seq, len_seq)
    )
    subsequent_mask = torch.triu(diagonal=1)  # 대각선 기준으로  위에가 0인상태대각은 모두 1

    return subsequent_mask
