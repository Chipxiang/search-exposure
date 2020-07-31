#!/usr/bin/env python
# coding: utf-8

# # network.py

# In[1]:


import torch
import torch.nn as nn

EMBED_SIZE = 768


class ENCODER(torch.nn.Module):

    def __init__(self):
        super(ENCODER, self).__init__()
        
        self.projection = nn.Linear(EMBED_SIZE, EMBED_SIZE)
        self.norm = nn.LayerNorm(EMBED_SIZE)
    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # data.py

# In[2]:


import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader


POS_NEG_PATH = "/datadrive/jianx/data/qidpidtriples.train.full.2.tsv"
QUERY_TRAIN_DICT_PATH = "/datadrive/jianx/data/queries.train.tsv"
PASSAGE_DICT_PATH = "/datadrive/jianx/data/collection.tsv"
TOP_DICT_PATH = "/datadrive/jianx/data/initial_ranking.dict"
RATING_DICT_PATH = "/datadrive/jianx/data/rel_scores.dict"
QUERY_TEST_DICT_PATH = "/datadrive/jianx/data/msmarco-test2019-queries.tsv"

NROW = None

def load_tsv_dict(path):
    with open(path) as file:
        line = file.readline()
        my_dict = {}
        while line:
            tokens = line.split("\t")
            indexid = int(tokens[0])
            text = tokens[1].rstrip()
            my_dict[indexid] = text
            line = file.readline()
    return my_dict


def load_pos_neg(path):
    data = pd.read_csv(path, sep='\t', header = None, nrows = NROW)
    return data

def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.loads(handle.read())

def load():
    pos_neg = load_pos_neg(POS_NEG_PATH)
    query_dict = load_tsv_dict(QUERY_TRAIN_DICT_PATH)
    passage_dict = load_tsv_dict(PASSAGE_DICT_PATH)
    top_dict = obj_reader(TOP_DICT_PATH)
    rating_dict = obj_reader(RATING_DICT_PATH)
    query_test_dict = load_tsv_dict(QUERY_TEST_DICT_PATH)
    return pos_neg, query_dict, passage_dict, top_dict, rating_dict, query_test_dict


# In[3]:


def encode_text(text, model, device):
    tokens = model.encode(text).to(device)
    if tokens.shape[0] > 512:
        tokens[:512]
    last_layer_features = model.extract_features(tokens)
    return last_layer_features[:,0,:]

class TrainDataset(Dataset):

    def __init__(self, pos_neg, queries, passages, model, device):

        self.pos_neg = pos_neg
        self.queries = queries
        self.passages = passages
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.pos_neg)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Positive negative pair
        qid = self.pos_neg.iloc[idx,0]
        pos_pid = self.pos_neg.iloc[idx,1]
        neg_pid = self.pos_neg.iloc[idx,2]
        self.model.train()
        q_tokens = encode_text(self.queries[qid], self.model, self.device)
        pos_tokens = encode_text(self.passages[pos_pid], self.model, self.device)
        neg_tokens = encode_text(self.passages[neg_pid], self.model, self.device)
        label = torch.tensor(0)
        sample = {'qid': qid, 'pos_pid': pos_pid, 'neg_pid': neg_pid, 'query': q_tokens,
                  'pos': pos_tokens, 'neg': neg_tokens, 'label': label}

        return sample


# # train.py

# In[4]:


def train(device, net, epochsize, dataiter, dataloader, optimizer):
    net.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = torch.tensor(0.0)
    for i in range(epochsize):
        try:
            batch = dataiter.next()
        except StopIteration:
            print("Finished iterating current dataset, begin reiterate")
            dataiter = iter(dataloader)
            batch = dataiter.next()
        queries = net(batch['query'].to(device))
        pos = net(batch['pos'].to(device)).to(device)
        neg = net(batch['neg'].to(device)).to(device)
        out_pos = (pos * queries).sum(1).to(device)
        out_neg = (neg * queries).sum(1).to(device)
        outputs = torch.cat((out_pos, out_neg), -1).to(device)
        labels = batch['label'].to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(i, loss.item())
    return train_loss / epochsize
        


# # test.py

# In[5]:


import torch
import random
import math
import numpy as np

def test_loader(device, net, model, top_dict, query_test_dict, passage_dict, rating_dict):
    net.eval()
    model.eval()
    qid_list = list(rating_dict.keys())
#     qid_list = random.sample(qid_list, test_batch)
    # sample test_batch of non-empty qids
    qids = []
    queries = []
    for qid in qid_list:
        qids.append(qid)
        queries.append(encode_text(query_test_dict[qid], model, device))
    result_dict = {}
    for i, qid in enumerate(qids):
        print(i)
        top_list = top_dict[qid]
        q_embed = net(queries[i].to(device)).detach()
        q_results = {}
        for j, pid in enumerate(top_list):
            p_seq = passage_dict[pid]
            p_embed = net(encode_text(p_seq, model, device)).detach().to(device)
            score = (p_embed * q_embed).sum(1).item()
            q_results[pid] = score
        result_dict[qid] = q_results
    #print_message(sorted(result_dict[qid].items(), key=lambda x: (x[1], [-1, 1][random.randrange(2)]), reverse=True))
    return result_dict


def get_ndcg_precision_rr(true_dict, test_dict, rank):
    sorted_result = sorted(test_dict.items(), key=lambda x: (x[1], [-1,1][random.randrange(2)]), reverse=True)
    original_rank = rank
    rank = min(rank, len(sorted_result))
    cumulative_gain = 0
    num_positive = 0
    rr = float("NaN")
    for i in range(len(sorted_result)):
        pid = sorted_result[i][0]
        if pid in true_dict:
            rr = 1 / (i + 1)
            break
    for i in range(rank):
        pid = sorted_result[i][0]
        if pid in true_dict:
            num_positive += 1
    sorted_result = sorted(test_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(rank):
        pid = sorted_result[i][0]
        relevance = 0
        if pid in true_dict:
            relevance = true_dict[pid]
        discounted_gain = relevance / math.log2(2 + i)
        cumulative_gain += discounted_gain
    sorted_ideal = sorted(true_dict.items(), key=lambda x: x[1], reverse=True)
    ideal_gain = 0
    for i in range(rank):
        relevance = 0
        if i < len(sorted_ideal):
            relevance = sorted_ideal[i][1]
        discounted_gain = relevance / math.log2(2 + i)
        ideal_gain += discounted_gain
    ndcg = 0
    if ideal_gain != 0:
         ndcg = cumulative_gain / ideal_gain
    return ndcg, num_positive / original_rank, rr


def test(device, net, model, top_dict, query_test_dict, passage_dict, rating_dict, rank):
    result_dict = test_loader(device, net, model, top_dict, query_test_dict, passage_dict, rating_dict)
    qids = list(result_dict.keys())
    result_ndcg = []
    result_prec = []
    result_rr = []
    for qid in qids:
        print(qid)
        ndcg, prec, rr = get_ndcg_precision_rr(rating_dict[qid], result_dict[qid], rank)
        result_ndcg.append(ndcg)
        result_prec.append(prec)
        result_rr.append(rr)
        print("qid: {} ndcg: {} prec: {} rr: {}".format(qid, ndcg, prec, rr))
    avg_ndcg = np.nanmean(result_ndcg)
    avg_prec = np.nanmean(result_prec)
    avg_rr = np.nanmean(result_rr)
    return avg_ndcg, avg_prec, avg_rr


# # main.py

# In[14]:


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import csv

BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCH_SIZE = 1500
NEPOCH = 5000
RANK= 10


# In[15]:


output_path = "./results/output_roberta_cut.csv"
roberta_model_path = "./results/roberta_model_cut.pt"
net_path = "./results/net_model_cut.pt"


# In[16]:


roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Current device is: %s" % torch.cuda.get_device_name(device))
roberta.cuda(device)
roberta.train()


# In[17]:


pos_neg, query_dict, passage_dict, top_dict, rating_dict, query_test_dict = load()


# In[18]:


trainset = TrainDataset(pos_neg, query_dict, passage_dict, roberta, device)
print(trainset.__len__())
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, 
                         shuffle=True, num_workers=0)
trainiter = iter(trainloader)


# In[19]:


net = ENCODER().to(device)


# In[20]:


optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# In[21]:


for i in range(NEPOCH):
    train_loss = train(device, net, EPOCH_SIZE, trainiter, trainloader, optimizer)
    avg_ndcg, avg_prec, avg_rr = test(device, net, roberta, top_dict, query_test_dict, passage_dict, rating_dict, rank = RANK)
    print("Epoch: {} {} {} {} {}".format(i, train_loss.item(), avg_ndcg, avg_prec, avg_rr))
    with open(output_path, mode='a+') as output:
        output_writer = csv.writer(output)
        output_writer.writerow([i, train_loss.item(), avg_ndcg, avg_prec, avg_rr])
    torch.save(net.state_dict(), net_path)
    torch.save(roberta.state_dict(), roberta_model_path)


# In[ ]:




