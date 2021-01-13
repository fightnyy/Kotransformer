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
import json

from os.path import dirname, abspath


import pdb
""" Current Working Directory """
CWD = os.getcwd()
parent = os.path.dirname(CWD)

"""config.json location"""
jsonpath = os.path.join(CWD, "config.json")

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
dataset_path = os.path.join(CWD, 'dataset')
print(dataset_path)
train_data = os.path.join(dataset_path, "ratings_train.json")
test_data = os.path.join(dataset_path , "ratings_test.json")
batch_size = 128
#pdb.set_trace()
"""sentencepiece model"""
vocab_file = os.path.join(dataset_path ,"kowiki.model")
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)



""" model saving """
#model_path = os.path.join(CWD,'model_checkpoints/')
#weight_PATH =model_path+weight_PATH
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


    @classmethod
    def load(cls, file):
         with open(file, 'r') as f:
            config = json.loads(f.read()) 
            return Config(config)   


"""config.json location"""
jsonpath = os.path.join(CWD, "config/config.json")

tconfig = Config.load(jsonpath)
