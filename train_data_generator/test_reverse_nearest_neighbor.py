import sys
sys.path.insert(0, '/home/jianx/search-exposure/forward_ranker/')
import torch
from train import generate_sparse
from load_data import obj_reader
import network
from annoy import AnnoyIndex

PASSAGE_INDEX = AnnoyIndex(EMBED_SIZE, 'euclidean')
PASSAGE_INDEX.load("/home/jianx/data/annoy/100_passage_index.ann")
PID_MAPPING = obj_reader("/home/jianx/data/annoy/100_pid_map.dict")


qids = []
queries = []
for key, value in QUERY_TEST_DICT.items():
    qid = key
    q_seq = value
    if q_seq != [] and qid not in qids:
        qids.append(qid)
        queries.append(generate_sparse(q_seq).to(DEVICE))
# compute cosine similarity
result_dict = {}
for i, qid in enumerate(qids):
    top_list = PASSAGE_INDEX.get_nns_by_vector(NET(generate_sparse(QUERY_TEST_DICT[qid]).to(DEVICE)).detach(),1000)
    print(top_list)
    for j, pid in enumerate(top_list):
        if pid in PID_MAPPING:
            top_list[j] = PID_MAPPING[pid]
        else:
            print("Not captured in mapping")
            top_list[j] = -1
    q_embed = NET(queries[i]).detach()
    q_results = {}
    score = float(len(top_list))
    for j, pid in enumerate(top_list):
        q_results[pid] = score
        score -= 1
    result_dict[qid] = q_results

#%%

brute_result = obj_reader("/home/jianx/data/results/brute_search_result.dict")

#%%

precision = 0
for qid, top_1000_list in brute_result.items():
    print(len(top_1000_list))
    annoy_top_1000 = result_dict[qid]
    precision += 1- len(set(top_1000_list) - set(annoy_top_1000))/len(top_1000_list)
precision /= len(brute_result)
print(precision)

#%%

import random
print(sorted(result_dict[qid].items(), key=lambda x: (x[1], [-1, 1][random.randrange(2)]), reverse=True))

#%%

from test import get_ndcg_precision_rr
import numpy as np
rating_dict = obj_reader("/home/jianx/data/rel_scores.dict")
rank = 10

qids = list(result_dict.keys())
result_ndcg = []
result_prec = []
result_rr = []
for qid in qids:
    if qid in rating_dict:
        ndcg, prec, rr = get_ndcg_precision_rr(rating_dict[qid], result_dict[qid], rank)
        result_ndcg.append(ndcg)
        result_prec.append(prec)
        result_rr.append(rr)
avg_ndcg = np.nanmean(result_ndcg)
avg_prec = np.nanmean(result_prec)
avg_rr = np.nanmean(result_rr)
print(avg_ndcg, avg_prec, avg_rr)