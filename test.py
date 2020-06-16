import torch
import random
import math
import numpy as np
from train import generate_sparse


def test_loader(net, test_batch, top_dict, query_test_dict, passage_dict, rating_dict):
    net.eval()
    device = net.device
    qid_list = list(rating_dict.keys())
    # sample test_batch of non-empty qids
    qids = []
    queries = []
    while len(qids) < test_batch:
        qid = random.sample(qid_list, 1)[0]
        q_seq = query_test_dict[qid]
        if q_seq != [] and qid not in qids:
            qids.append(qid)
            queries.append(generate_sparse(q_seq).to(device))
    # compute cosine similarity
    result_dict = {}
    for i, qid in enumerate(qids):
        top_list = top_dict[qid]
        q_embed = net(queries[i])
        q_results = {}
        for j, pid in enumerate(top_list):
            p_seq = passage_dict[pid]
            if not p_seq:
                score = -1
            else:
                p_embed = net(generate_sparse(p_seq).to(device))
                score = torch.cosine_similarity(q_embed.unsqueeze(0), p_embed.unsqueeze(0)).item()
            q_results[pid] = score
        result_dict[qid] = q_results
    return result_dict


def get_ndcg_precision_rr(true_dict, test_dict, rank):
    sorted_result = sorted(test_dict.items(), key=lambda x: x[1], reverse=True)
    rank = min(rank, len(sorted_result))
    cumulative_gain = 0
    ideal_dict = {}
    num_positive = 0
    rr = float("NaN")
    for i in range(len(sorted_result)):
        pid = sorted_result[i][0]
        if pid in true_dict:
            rr = 1 / (i + 1)
            break

    for i in range(rank):
        pid = sorted_result[i][0]
        relevance = 0
        if pid in true_dict:
            relevance = true_dict[pid]
            num_positive += 1
        ideal_dict[pid] = relevance
        discounted_gain = (2 ** relevance - 1) / math.log2(2 + i)
        cumulative_gain += discounted_gain
    sorted_ideal = sorted(ideal_dict.items(), key=lambda x: x[1], reverse=True)
    ideal_gain = 0
    for i in range(rank):
        relevance = sorted_ideal[i][1]
        discounted_gain = (2 ** relevance - 1) / math.log2(2 + i)
        ideal_gain += discounted_gain
    ndcg = float("NaN")
    if ideal_gain != 0:
        ndcg = cumulative_gain / ideal_gain
    return ndcg, num_positive / rank, rr


def test(net, test_batch, top_dict, query_test_dict, passage_dict, rating_dict, rank):
    result_dict = test_loader(net, test_batch, top_dict, query_test_dict, passage_dict, rating_dict)
    qids = list(result_dict.keys())
    result_ndcg = []
    result_prec = []
    result_rr = []
    for qid in qids:
        true_dict = rating_dict[qid]
        test_dict = result_dict[qid]
        ndcg, prec, rr = get_ndcg_precision_rr(true_dict, test_dict, rank)
        result_ndcg.append(ndcg)
        result_prec.append(prec)
        result_rr.append(rr)
    avg_ndcg = np.nanmean(result_ndcg)
    avg_prec = np.nanmean(result_prec)
    avg_rr = np.nanmean(result_rr)
    return avg_ndcg, avg_prec, avg_rr
