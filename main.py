# -*- coding: utf-8 -*-

from transformer import Transformer
from config import config
from moviedataset import MovieDataset, movie_collate_fn
from tqdm import tqdm
from movieclassifier import MovieClassification


import torch
import torch.nn as nn
import sentencepiece as spm
import numpy as np

batch_size = config.batch_size
vocab = config.vocab

dataset_path = config.dataset_path
train_data = config.train_data
test_data = config.test_data

train_dataset = MovieDataset(vocab, train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = movie_collate_fn)

test_dataset = MovieDataset(vocab, test_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = movie_collate_fn)


"""
모델 평가 epoch
"""

def eval_epoch(config, model, data_loader):
    matches = []
    model.eval()
    
    n_word_total =  0
    n_correct_total = 0
    with tqdm(total = len(data_loader), desc = f"Valid") as pbar:
        for i, value in enumerate(data_loader):
              labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

              outputs = model.forward(enc_inputs, dec_inputs)
              """
              #logits.size = [batch_size, n_outputs]
              """
              logits = outputs[0]
              """
              #max를 쓴 이유는  dim =1 즉 negative, positive 중에 어떤 것을 더 선호하는것인지 보려고
              """
              _, indices = logits.max(1)

              match = torch.eq(indices, labels).detach()
              matches.extend(match.to("cpu"))
              # A if condition else B
              # condition 참이면 A가 나오고 False 면 else 가 나온 
              accuracy = np.sum(matches) / len(matches) if 0 <len(matchs) else 0
              
              pbar.update(1)
              pbar.set_postfix_str(f"Acc: {accuracy:.3f}")

    return np.sum(matchs)/len(matchs) if 0 < len(matches) else 0

"""
모델 학습 epoch
"""
def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
              labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

              optimizer.zero_grad()
              outputs = model.forward(enc_inputs, dec_inputs)
              logits = outputs[0]

              loss = criterion(logits, labels)
              loss_val = loss.item()
              losses.append(loss_val)

              loss.backward()
              optimizer.step()

              pbar.update(1)
              pbar.set_postfix_str(f"Loss: {loss_val:.3f}({np.mean(losses):.3f})")
    return np.mean(losses)

if __name__ == '__main__':
    config.tconfig.device = config.device
    n_epoch = config.tconfig.n_epoch
    print("="*10)
    print("config : ",config.tconfig)
    print("="*10)
    
   
    model = MovieClassification(config.tconfig)
    model = nn.DataParallel(model)
    model.to(config.tconfig.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.tconfig.learning_rate)

    losses, scores = [],[]
    for epoch in range(n_epoch):
        loss = train_epoch(config.tconfig, epoch, model, criterion, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        losses.append(loss)
        scores.append(score)


