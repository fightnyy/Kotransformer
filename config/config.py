#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import datetime

import torch
import torch.nn as nn
import sentencepiece as spm



""" Current Working Directory """
CWD = os.getcwd()

""" cuda device """
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


""" logger """
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
now = datetime.datetime.now()
today = '%s-%s-%s' % (now.year, now.month, now.day)
current_time = '%s-%s-%s' % (now.hour, now.minute, now.second)
log_path = 'log/' + today


""" dataset """
dataset_path = os.path.join(CWD, 'data')
batch_size = 32

"""sentencepiece model"""
vocab_file = os.path.join(CWD,"dataset/kowiki.model")
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)



""" model saving """
#model_path = os.path.join(CWD,'model_checkpoints/')
#weight_PATH =model_path+weight_PATH
