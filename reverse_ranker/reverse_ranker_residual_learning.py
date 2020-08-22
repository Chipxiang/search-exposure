#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Load data: passage_embeddings, query_train_embeddings, train_data (a proportion of)
# 2. Network input: 768 output: 32
# 3. Train: concatenate original 768 with 32 


# ## Import data: load_data.py

# In[1]:


import csv
import pickle
import numpy as np

PASSAGE_NP_PATH = "/home/jianx/results/passage_0__emb_p__data_obj_0.pb"
PASSAGE_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_passage_map.dict"
QUERY_TRAIN_NP_PATH = "/home/jianx/results/query_0__emb_p__data_obj_0.pb"
QUERY_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_query_train_map.dict"
TRAIN_RANK_PATH = "/datadrive/jianx/data/train_data/ance_training_rank100_200000_random.csv"
# TRAIN_RANK_PATH = "/datadrive/ruohan/reverse_ranker/new_training/combine_rank_train_phase2.csv"


OUT_RANK = 200
N_PASSAGE = 200000
def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")
def load_train(path):
    with open(path, "r") as file:
        pos_dict = {}
        neg_dict = {}
        count = 0
        for line in file:
#             if count >= N_PASSAGE * 100:
#                 break
#             count += 1
            tokens = line.split(",")
            pid = int(tokens[0])
            qid = int(tokens[1])
            rank = int(tokens[2].rstrip())
            if rank == 0:
                if pid not in neg_dict:
                    neg_dict[pid] = {}
                neg_dict[pid][qid] = OUT_RANK
            else:
                if pid not in pos_dict:
                    pos_dict[pid] = {}
                pos_dict[pid][qid] = rank
    return pos_dict, neg_dict
def map_id(old_np, mapping):
    new_dict = dict(zip(mapping.values(),old_np))
    return new_dict
def load():
    print("Load embeddings.")
    passage_np = obj_reader(PASSAGE_NP_PATH)
    pid_mapping = obj_reader(PASSAGE_MAP_PATH)
    query_np = obj_reader(QUERY_TRAIN_NP_PATH)
    qid_mapping = obj_reader(QUERY_MAP_PATH)
    print("Mapping ids.")
    query_dict = map_id(query_np, qid_mapping)
    passage_dict = map_id(passage_np, pid_mapping)
    print("Load training data.")
    train_pos_dict, train_neg_dict = load_train(TRAIN_RANK_PATH)
    return train_pos_dict, train_neg_dict, query_dict, passage_dict


# In[2]:


train_pos_dict, train_neg_dict, query_dict, passage_dict = load()


# ## Network Architecture: network.py

# In[10]:


import torch
import torch.nn as nn

NUM_HIDDEN_NODES = 1536
NUM_HIDDEN_LAYERS = 15
DROPOUT_RATE = 0.1
    
# Define the network
class ResidualNet(torch.nn.Module):

    def __init__(self, embed_size):
        super(ResidualNet, self).__init__()
        
        self.input = nn.Linear(embed_size, NUM_HIDDEN_NODES)
        self.relu = nn.ReLU()
        self.normlayer = nn.LayerNorm(NUM_HIDDEN_NODES)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.output = nn.Linear(NUM_HIDDEN_NODES, embed_size)

    def forward(self, x):
        identity = x
        out = x
        for i in range(NUM_HIDDEN_LAYERS):
            out = self.input(out)
            out = self.relu(out)
            out = self.normlayer(out)
            out = self.dropout(out)
            out = self.output(out)
            out += identity
            out = self.relu(out)
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ## Train reverse ranker: train.py

# In[15]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F


TOP_K = 100

# With probability alpha
# Select a random negative sample from train_neg_dict
ALPHA = 0.5

def dot_product(A, B, normalize=False):
    if normalize:
        A = F.normalize(A)
        B = F.normalize(B)
    b = A.shape[0]
    embed = A.shape[1]
    result = torch.bmm(A.view(b, 1, embed), B.view(b, embed, 1))
    return result


def mini_batch(batch_size, device, train_pos_dict, train_neg_dict, query_dict, passage_dict):
    passage_list = list(train_neg_dict.keys())
    passages = []
    pos = []
    neg = []
    pos_rank_list = []
    neg_rank_list = []
    while len(passages) < batch_size:
        pid = random.sample(passage_list, 1)[0]
        try:
            temp_pos_list = list(train_pos_dict[pid].keys())
        except:
            continue
        try:
            temp_neg_list = list(train_neg_dict[pid].keys())
        except:
            continue
        if np.random.uniform(0,1,1) <= ALPHA:
            random_positive = random.sample(temp_pos_list, 1)
            pos_qid = random_positive[0]
            pos_rank = train_pos_dict[pid][pos_qid]
            random_negative = random.sample(temp_neg_list, 1)
            neg_qid = random_negative[0]
            neg_rank = train_neg_dict[pid][neg_qid]
#             not_negative = True
#             while not_negative:
#                 temp_neg_qid = random.sample(list(query_dict.keys()), 1)
#                 if temp_neg_qid not in temp_query_list:
#                     neg_qid = temp_neg_qid[0]
#                     neg_rank = 1000
#                     not_negative = False
        else:
            if len(temp_pos_list) < 2:
                continue
            pos_neg_pair = random.sample(temp_pos_list, 2)
            # e.g. 60 >= 3
            if train_pos_dict[pid][pos_neg_pair[0]] >= train_pos_dict[pid][pos_neg_pair[1]]:
                pos_qid = pos_neg_pair[1]
                neg_qid = pos_neg_pair[0]
            # e.g. 3 < 60
            else:
                pos_qid = pos_neg_pair[0]
                neg_qid = pos_neg_pair[1]   
            pos_rank = train_pos_dict[pid][pos_qid]
            neg_rank = train_pos_dict[pid][neg_qid]
        p_seq = passage_dict[pid]
        pos_seq = query_dict[pos_qid]
        neg_seq = query_dict[neg_qid]
        passages.append(p_seq)
        pos.append(pos_seq)
        neg.append(neg_seq)
        pos_rank_list.append(TOP_K - pos_rank)
        neg_rank_list.append(TOP_K - neg_rank)
#         pos_rank_list.append((TOP_K - pos_rank) * 2)
#         neg_rank_list.append((TOP_K - neg_rank) * 2)
    labels = torch.stack([torch.FloatTensor(pos_rank_list), torch.FloatTensor(neg_rank_list)], dim=1)
    passages = torch.from_numpy(np.stack(passages))
    pos = torch.from_numpy(np.stack(pos))
    neg = torch.from_numpy(np.stack(neg))
    return passages.to(device), pos.to(device), neg.to(device), labels.to(device)


def train(net, epoch_size, batch_size, optimizer, device, train_pos_dict, train_neg_dict, 
          query_dict, passage_dict, scale=10, loss_option="ce"):
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    train_loss = 0.0
    net.train()
    for mb_idx in range(epoch_size):
        # Read in a new mini-batch of data!
        passages, pos, neg, labels = mini_batch(batch_size, device, train_pos_dict, train_neg_dict, 
                                                query_dict, passage_dict)
        optimizer.zero_grad()
        p_embed = net(passages).to(device)
        pos_embed = net(pos).to(device)
        neg_embed = net(neg).to(device)
        out_pos = dot_product(p_embed, pos_embed).to(device)
        out_neg = dot_product(p_embed, neg_embed).to(device)
        out = torch.cat((out_pos, out_neg), -1).squeeze()
        if loss_option == "bce":
            loss = bce(softmax(out), softmax(labels))
        if loss_option == "ce":
            loss = ce(out, torch.tensor([0 for i in range(batch_size)]).to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(str(mb_idx) + " iteration: " + str(train_loss / (mb_idx + 1)))
    return train_loss / epoch_size


# ## Main function: main.py

# In[16]:


import datetime
from datetime import datetime, timezone, timedelta

TIME_OFFSET = -4


def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedelta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)


# In[17]:


import torch
from torch import optim
import csv
import sys
import os


MODEL_PATH = "/datadrive/ruohan/random_sample/"
CURRENT_DEVICE = "cuda:0"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path, embed_size, pretrained=False):
    if pretrained:
        net = ResidualNet(embed_size=embed_size)
        net.load_state_dict(torch.load("/home/ruohan/DSSM/search-exposure/reverse_ranker/results/reverse_fine_tune1000_100_1000_0.0001_32.model"))
        net.to(CURRENT_DEVICE)
    else:
        net = ResidualNet(embed_size=embed_size).to(CURRENT_DEVICE)
    print("Loading data")
#     train_rank_dict, query_dict, passage_dict = load()
#     print("Data successfully loaded.")
#     print("Positive Negative Pair dict size: " + str(len(train_rank_dict)))
#     print("Num of queries: " + str(len(query_dict)))
#     print("Num of passages: " + str(len(passage_dict)))
#     print("Finish loading.")

    arg_str = "reverse_alpha0.5_initial_residual_saveoptim_layer15" + str(num_epochs) + "_" + str(epoch_size) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(
        embed_size)
    unique_path = model_path + arg_str + ".model"
    output_path = model_path + arg_str + ".csv"
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for ep_idx in range(num_epochs):
        train_loss = train(net, epoch_size, batch_size, optimizer, CURRENT_DEVICE, train_pos_dict, 
                           train_neg_dict, query_dict, passage_dict)
        print_message([ep_idx,train_loss])
        with open(output_path, mode='a+') as output:
            output_writer = csv.writer(output)
            output_writer.writerow([ep_idx, train_loss])
        torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_epoch": ep_idx,
                "train_loss": train_loss,
                "n_hidden_layers": NUM_HIDDEN_LAYERS
                    }, unique_path)


# In[ ]:


# Train on randomly sampled training data that contains 200,000 passages
main(1000,100,1000,0.0001,MODEL_PATH,768)


# In[ ]:




