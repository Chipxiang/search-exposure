#!/usr/bin/env python
# coding: utf-8

# ## Import data: load_data.py

# In[28]:


import csv
import pickle


PASSAGE_DICT_PATH = "/home/jianx/data/passages.dict"
QUERY_TRAIN_DICT_PATH = "/home/jianx/data/queries_train.dict"
TRAIN_RANK_PATH = "/home/jianx/data/train_data/256_200000_100_100_training.csv"
def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")
def load_train(path):
    with open(path) as file:
        line = file.readline()
        my_dict = {}
        while line:
            tokens = line.split(",")
            pid = int(tokens[0])
            qid = int(tokens[1])
            rank = int(tokens[2].rstrip())
            if pid not in my_dict:
                my_dict[pid] = {}
            my_dict[pid][qid] = rank
            line = file.readline()
    return my_dict
def load():
    query_dict = obj_reader(QUERY_TRAIN_DICT_PATH)
    passage_dict = obj_reader(PASSAGE_DICT_PATH)
    train_rank_dict = load_train(TRAIN_RANK_PATH)
    return train_rank_dict, query_dict, passage_dict


# ## Network Architecture: network.py

# In[29]:


import torch
import torch.nn as nn

NUM_HIDDEN_NODES = 64
NUM_HIDDEN_LAYERS = 3
DROPOUT_RATE = 0.1
FEAT_COUNT = 100000


# Define the network
class DSSM(torch.nn.Module):

    def __init__(self, embed_size):
        super(DSSM, self).__init__()

        layers = []
        last_dim = FEAT_COUNT
        for i in range(NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(last_dim, NUM_HIDDEN_NODES))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(NUM_HIDDEN_NODES))
            layers.append(nn.Dropout(p=DROPOUT_RATE))
            last_dim = NUM_HIDDEN_NODES
        layers.append(nn.Linear(last_dim, embed_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ## Train reverse ranker: train.py

# In[30]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

VOCAB_LEN = 100000
TOP_K = 100

# With probability alpha
# Select a random negative sample from rest of the queries
ALPHA = 0.8

def generate_sparse(idx, vocab_len=VOCAB_LEN):
    index_tensor = torch.LongTensor([idx])
    value_tensor = torch.Tensor([1/len(idx)] * len(idx))
    sparse_tensor = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([vocab_len, ]))
    return sparse_tensor


def mini_batch(batch_size, device, train_rank_dict, query_dict, passage_dict):
    passage_list = list(train_rank_dict.keys())
    passages = []
    pos = []
    neg = []
    pos_rank_list = []
    neg_rank_list = []
    while len(passages) < batch_size:
        pid = random.sample(passage_list, 1)[0]
        temp_query_list = list(train_rank_dict[pid].keys())
        if np.random.uniform(0,1,1) <= ALPHA:
            random_positive = random.sample(temp_query_list, 1)
            pos_qid = random_positive[0]
            pos_rank = 1
            not_negative = True
            while not_negative:
                temp_neg_qid = random.sample(list(query_dict.keys()), 1)
                if temp_neg_qid not in temp_query_list:
                    neg_qid = temp_neg_qid[0]
                    neg_rank = 1000
                    not_negative = False
        else:
            if len(temp_query_list) < 2:
                continue
            pos_neg_pair = random.sample(temp_query_list, 2)
            if train_rank_dict[pid][pos_neg_pair[0]] >= train_rank_dict[pid][pos_neg_pair[1]]:
                pos_qid = pos_neg_pair[0]
                neg_qid = pos_neg_pair[1]
                pos_rank = train_rank_dict[pid][pos_neg_pair[0]]
                neg_rank = train_rank_dict[pid][pos_neg_pair[1]]
            else:
                pos_qid = pos_neg_pair[1]
                neg_qid = pos_neg_pair[0]   
                pos_rank = train_rank_dict[pid][pos_neg_pair[1]]
                neg_rank = train_rank_dict[pid][pos_neg_pair[0]]
        p_seq = passage_dict[pid]
        pos_seq = query_dict[pos_qid]
        neg_seq = query_dict[neg_qid]
        if p_seq != [] and pos_seq != [] and neg_seq != []:
            passages.append(generate_sparse(p_seq))
            pos.append(generate_sparse(pos_seq))
            neg.append(generate_sparse(neg_seq))
#             pos_rank_list.append(TOP_K - pos_rank)
#             neg_rank_list.append(TOP_K - neg_rank)
            pos_rank_list.append((TOP_K - pos_rank) * 2)
            neg_rank_list.append((TOP_K - neg_rank) * 2)
    labels = torch.stack([torch.FloatTensor(pos_rank_list), torch.FloatTensor(neg_rank_list)], dim=1)
    return torch.stack(passages).to(device), torch.stack(pos).to(device), torch.stack(neg).to(device), labels.to(device)


def train(net, epoch_size, batch_size, optimizer, device, train_rank_dict, query_dict,
          passage_dict, scale):
    criterion = nn.BCELoss()
    softmax = nn.Softmax(dim=1)
    train_loss = 0.0
    net.train()
    for mb_idx in range(epoch_size):
        # Read in a new mini-batch of data!
        passages, pos, neg, labels = mini_batch(batch_size, device, train_rank_dict, query_dict,
                                               passage_dict)
        optimizer.zero_grad()
        p_embed = net(passages)
        pos_embed = net(pos)
        neg_embed = net(neg)
        out_pos = torch.cosine_similarity(p_embed, pos_embed).unsqueeze(0).T
        out_neg = torch.cosine_similarity(p_embed, neg_embed).unsqueeze(0).T
        out = torch.cat((out_pos, out_neg), -1) * torch.tensor([scale], dtype=torch.float).to(device)
        loss = criterion(softmax(out), softmax(labels))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(str(mb_idx) + " iteration: " + str(train_loss / (mb_idx + 1)))
    return train_loss / epoch_size


# ## Main function: main.py

# In[31]:


import torch
from torch import optim
import csv
import sys
import os

# NUM_EPOCHS = int(sys.argv[1])
# EPOCH_SIZE = int(sys.argv[2])
# BATCH_SIZE = int(sys.argv[3])
# LEARNING_RATE = float(sys.argv[4])
# EMBED_SIZE = int(sys.argv[5])
# SCALE = int(sys.argv[6])
# GPU_ROOT = "/home/jianx/data/gpu_usage.list"

# CURRENT_GPU_ID, CURRENT_DEVICE = select_device(GPU_ROOT)
# print(CURRENT_DEVICE)
# print("Num of epochs:", NUM_EPOCHS)
# print("Epoch size:", EPOCH_SIZE)
# print("Batch size:", BATCH_SIZE)
# print("Learning rate:", LEARNING_RATE)
# print("Embedding size:", EMBED_SIZE)
# print("Scale size:", SCALE)
RANK = 10
TEST_BATCH = 43
MODEL_PATH = "./results/"
FORWARD_RANKER_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path, rank, test_batch, embed_size, scale, 
         pretrained_option=True):
    if pretrained_option:
        net = DSSM(embed_size=embed_size)
        net.load_state_dict(torch.load(FORWARD_RANKER_PATH))
        net.to(CURRENT_DEVICE)
    else:
        net = DSSM(embed_size=embed_size).to(CURRENT_DEVICE)
    print("Loading data")
    train_rank_dict, query_dict, passage_dict = load()
    print("Data successfully loaded.")
    print("Positive Negative Pair dict size: " + str(len(train_rank_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))
    print("Finish loading.")

    arg_str = "200000_samples_0.8_neg" + str(num_epochs) + "_" + str(epoch_size) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(
        embed_size) + "_" + str(scale)
    unique_path = model_path + arg_str + ".model"
    output_path = model_path + arg_str + ".csv"
    for ep_idx in range(num_epochs):
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        train_loss = train(net, epoch_size, batch_size, optimizer, CURRENT_DEVICE, train_rank_dict,
                           query_dict, passage_dict, scale)
        print(ep_idx,train_loss)
#         avg_ndcg, avg_prec, avg_rr = test(net, CURRENT_DEVICE, test_batch, top_dict, query_test_dict, passage_dict,
#                                           rating_dict, rank)
#         print("Epoch:{}, loss:{}, NDCG:{}, P:{}, RR:{}".format(ep_idx, train_loss, avg_ndcg, avg_prec, avg_rr))
        with open(output_path, mode='a+') as output:
            output_writer = csv.writer(output)
            output_writer.writerow([ep_idx, train_loss])
        torch.save(net.state_dict(), unique_path)
#     cleanup_gpu_list(CURRENT_GPU_ID, GPU_ROOT)


# if __name__ == '__main__':
#     main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, RANK, TEST_BATCH, EMBED_SIZE, SCALE)


# In[32]:


CURRENT_DEVICE = "cuda:0"


# In[ ]:


main(200,10,500,0.001,MODEL_PATH,RANK,TEST_BATCH,256,10)


# In[ ]:




