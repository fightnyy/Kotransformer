#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import datasetconfig
import os
import sys
import csv

scriptPATH = "../config/"
sys.path.append(os.path.abspath(scriptPATH))
import config

csv.field_size_limit(sys.maxsize)
logger = config.logger
in_file = os.path.join(datasetconfig.sentencepath, "kowiki_20201230.csv")
out_file = "kowiki.txt"
seperator = "␝"
if os.path.exists(out_file):
    print("kowiki.txt is already exist")
else:
    df = pd.read_csv(in_file, sep=seperator, engine="python")
    with open(out_file, "w") as f:
        for index, row in df.iterrows():
            f.write(row["text"])  # title 과 text 가 중복되어 하나만 저장
            f.write("\n\n\n\n")  # 구분자
