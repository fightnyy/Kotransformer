# -*- coding: utf-8 -*-

from transformer import Transformer
from config import config
from moviedataset import MovieDataset, movie_collate_fn, build_data_loader
from tqdm import tqdm, trange
from movieclassifier import MovieClassification
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import sentencepiece as spm
import numpy as np
import argparse

batch_size = config.batch_size
vocab = config.vocab

dataset_path = config.dataset_path
train_data = config.train_data
test_data = config.test_data



"""init process group"""
def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank = rank,world_size =world_size)

"""destory process group"""
def destroy_process_group():
    dist.init_process_group()

"""
모델 평가 epoch
"""

def eval_epoch(config, rank, model, data_loader):
    matches = []
    model.eval()
    
    n_word_total =  0
    n_correct_total = 0
    with tqdm(total = len(data_loader), desc = f"Valid{rank}") as pbar:
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
              #numpy는 CPU에서 연산을 해주어야 하기 때문에 match를 GPU에서 detach 해준 것임
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
def train_epoch(config, rank, epoch, model, criterion, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train model using {rank} GPU in {epoch} epoch config.device is = {config.device}") as pbar:
        for i, value in enumerate(train_loader):
              labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

              optimizer.zero_grad()
              #outputs = logits, enc_self_attn_probs, self_attn_probs, dec_enc_attn_prbs
              outputs = model.forward(enc_inputs, dec_inputs)
              logits = outputs[0]#이렇게 하면 logits 만 받아옴 

              loss = criterion(logits, labels)
              loss_val = loss.item()
              losses.append(loss_val)

              loss.backward()
              optimizer.step()
              scheduler.step()

              pbar.update(1)
              pbar.set_postfix_str(f"Loss: {loss_val:.3f}({np.mean(losses):.3f})")
    return np.mean(losses)


def train_model(rank, world_size, args):

    if 1 < args.n_gpu:
        init_process_group(rank, world_size)
    master = (world_size == 0 or rank % world_size ==0)#아예 GPU가 없거나 아니면 0번째 GPU 일떄
    
    config.tconfig.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config.tconfig)

    best_epoch, best_loss, best_score = 0, 0, 0
    model = MovieClassification(config.tconfig)
    
    if os.path.isfile(args.save):
        print(f"rank:{rank} load state dict from : {args.save}")
    if 1 < args.n_gpu:
        model.to(config.tconfig.device)
        model = DistributedDataParallel(model, device_ids = [rank], find_unused_parameters = True)

    else : # GPU가 1개이면 하나에만 넣는것 또는 GPU 가 없으면 CPU에 넣는것 
        model.to(config.tconfig.device)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader, train_sampler = build_data_loader(vocab, train_data, args, shuffle = True)
    test_loader, _ = build_data_loader(vocab, test_data, args, shuffle = False)

    t_total = len(train_loader) * config.tconfig.n_epoch
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':config.tconfig.weight_decay},
        {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = config.tconfig.learning_rate, eps = config.tconfig.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps =config.tconfig.warmup_steps, num_training_steps = t_total)
    
    offset = best_epoch
    for step in trange(config.tconfig.n_epoch, desc="Epoch"):
        if train_sampler :
            train_sampler.set_epoch(step)
        epoch = step + offset

        loss = train_epoch(config.tconfig, rank, epoch, model, criterion, optimizer, scheduler, train_loader)
        score = eval_epoch(config.tconfig, rank, model, test_loader)

        if master and best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
            if isinstance(model, DistributedDataParallel):#isinstance(a:object,b:classtype) => 만약 a가 b의 subclass instance 이거나 그냥 instance 인경우는 True
                model.module.save(best_epoch, best_loss, best_score, args.save)
            else:
                model.save(best_epoch, best_loss, best_score, args.save)
            print(f">>>> rank: {rank} save model to {args.save}, epoch = {best_epocs}, loss = {best_loss:.3f}, score = {best_score:.3f}")


    if 1<args.n_gpu:
        destroy_process_group()
"""
rank = GPU ID를 나타냄 예를들어 GPU가 4개면 첫번째 GPU의 rank 는 0이고 rank는 유니크한 값이다.

world_Size = nprocs = args.n_gpu = 컴퓨터에 있는 GPU 갯수
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default = None, type = int, required = False)
    parser.add_argument("--save", default="save_best.pth", type=str, required = False, help = "path of the save file")
    args = parser.parse_args()
    config.tconfig.device = config.device
    
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0
    print("available GPU : ",args.n_gpu)
    if 1 < args.n_gpu:
        mp.spawn(train_model,
             args=(args.n_gpu, args),
             nprocs=args.n_gpu,
             join=True)
    else:
        train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)




