import torch
import torch.nn as nn
import torch.optim as optim
import random

from load_data import load



def sparse_to_dense(idx, vocab_len = 10):
    index_tensor = torch.LongTensor([idx])
    value_tensor = torch.Tensor([1]*len(idx))
    dense_tensor = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([vocab_len,])).to_dense()
    return dense_tensor

def mini_batch(batch_size):
	positive_dict, negative_dict, query_dict, passage_dict = load()
    query_list = list(positive_dict.keys())
    qids = []
    queries = []
    pos = []
    neg = []
    while len(qids) < batch_size:
        qid = random.sample(query_list, 1)[0]
        pos_pid = positive_dict[qid]
        neg_pid = random.sample(negative_dict[qid], 1)[0]
        q_seq = query_dict[qid]
        pos_seq = passage_dict[pos_pid]
        neg_seq = passage_dict[neg_pid]
        if q_seq != [] and pos_seq != [] and neg_seq != []:
            qids.append(qid)
            queries.append(sparse_to_dense(q_seq))
            pos.append(sparse_to_dense(pos_seq))
            neg.append(sparse_to_dense(neg_seq))
    labels = [1 for i in range(batch_size)]
    return torch.stack(queries).to(DEVICE), torch.stack(pos).to(DEVICE), torch.stack(neg).to(DEVICE), labels

def train(net, EPOCH_SIZE):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
	train_loss = 0.0
	net.train()
	for mb_idx in range(EPOCH_SIZE):
	    #Read in a new mini-batch of data!
	    queries, pos, neg, labels = mini_batch(2)
	    q_embed = net(queries)
	    pos_embed = net(pos)
	    neg_embed = net(neg)
	    out_pos = torch.cosine_similarity(q_embed, pos_embed).unsqueeze(0).T
	    out_neg = torch.cosine_similarity(q_embed, neg_embed).unsqueeze(0).T
	    out = torch.cat((out_pos,out_neg), -1)
	    loss = criterion(out, torch.tensor(labels).to(DEVICE))
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	    train_loss += loss.item()
	return train_loss / EPOCH_SIZE