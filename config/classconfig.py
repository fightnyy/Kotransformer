#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


    @classmethod
    def load(cls, file):
         with open(file, 'r') as f:
            config = json.loads(f.read()) 
            return Config(config)   


"""config.json location"""
jsonpath = os.path.join(os.getcwd(), "config.json")

path = jsonpath
config = Config.load(path)
