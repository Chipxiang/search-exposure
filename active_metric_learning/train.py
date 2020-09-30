import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F



# With probability alpha
# Select a random negative sample from train_neg_dict

def dot_product(A, B, normalize=False):
    if normalize:
        A = F.normalize(A)
        B = F.normalize(B)
    b = A.shape[0]
    embed = A.shape[1]
    result = torch.bmm(A.view(b, 1, embed), B.view(b, embed, 1))
    return result


def mini_batch(opts, train_pos_dict, train_neg_dict, query_dict, passage_dict):
    batch_size = opts.batch_size
    device = opts.device
    alpha = opts.alpha
    top_k = opts.top_k
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
        if np.random.uniform(0,1,1) <= alpha:
            random_positive = random.sample(temp_pos_list, 1)
            pos_qid = random_positive[0]
            pos_rank = train_pos_dict[pid][pos_qid]
            random_negative = random.sample(temp_neg_list, 1)
            neg_qid = random_negative[0]
            neg_rank = train_neg_dict[pid][neg_qid]
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
        pos_rank_list.append(top_k - pos_rank)
        neg_rank_list.append(top_k - neg_rank)
    labels = torch.stack([torch.FloatTensor(pos_rank_list), torch.FloatTensor(neg_rank_list)], dim=1)
    passages = torch.from_numpy(np.stack(passages))
    pos = torch.from_numpy(np.stack(pos))
    neg = torch.from_numpy(np.stack(neg))
    return passages.to(device), pos.to(device), neg.to(device), labels.to(device)


def train(net, optimizer, opts, train_pos_dict, train_neg_dict, 
          query_dict, passage_dict, loss_option="ce"):
    epoch_size = opts.epoch_size
    batch_size = opts.batch_size
    device = opts.device
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    train_loss = 0.0
    net.train()
    for mb_idx in range(epoch_size):
        # Read in a new mini-batch of data!
        passages, pos, neg, labels = mini_batch(opts, train_pos_dict, train_neg_dict, 
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
    return train_loss / epoch_size
