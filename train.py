import torch
import torch.nn as nn
import torch.optim as optim
import random

VOCAB_LEN = 50000


def sparse_to_dense(idx, vocab_len=VOCAB_LEN):
    index_tensor = torch.LongTensor([idx])
    value_tensor = torch.Tensor([1] * len(idx))
    dense_tensor = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([vocab_len, ])).to_dense()
    return dense_tensor


def mini_batch(batch_size, device, positive_dict, negative_dict, query_dict, passage_dict):
    query_list = list(positive_dict.keys())
    queries = []
    pos = []
    neg = []
    while len(queries) < batch_size:
        qid = random.sample(query_list, 1)[0]
        pos_pid = positive_dict[qid]
        neg_pid = random.sample(negative_dict[qid], 1)[0]
        q_seq = query_dict[qid]
        pos_seq = passage_dict[pos_pid]
        neg_seq = passage_dict[neg_pid]
        if q_seq != [] and pos_seq != [] and neg_seq != []:
            queries.append(sparse_to_dense(q_seq))
            pos.append(sparse_to_dense(pos_seq))
            neg.append(sparse_to_dense(neg_seq))
    labels = [1 for i in range(batch_size)]
    return torch.stack(queries).to(device), torch.stack(pos).to(device), torch.stack(neg).to(device), labels


def train(net, epoch_size, batch_size, learning_rate, device, positive_dict, negative_dict, query_dict,
          passage_dict):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_loss = 0.0
    net.train()

    for mb_idx in range(epoch_size):
        # Read in a new mini-batch of data!
        queries, pos, neg, labels = mini_batch(batch_size, device, positive_dict, negative_dict, query_dict,
                                               passage_dict)
        q_embed = net(queries)
        pos_embed = net(pos)
        neg_embed = net(neg)
        out_pos = torch.cosine_similarity(q_embed, pos_embed).unsqueeze(0).T
        out_neg = torch.cosine_similarity(q_embed, neg_embed).unsqueeze(0).T
        out = torch.cat((out_pos, out_neg), -1)
        loss = criterion(out, torch.tensor(labels).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print(str(mb_idx) + " iteration: " + str(train_loss / (mb_idx + 1)))
    return train_loss / epoch_size
