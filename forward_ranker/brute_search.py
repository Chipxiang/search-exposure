import torch
from train import generate_sparse
from load_data import obj_reader
from load_data import obj_writer
from test import get_ndcg_precision_rr
import network
import numpy as np
import datetime
from datetime import datetime, timezone, timedelta

TIME_OFFSET = -4


def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedelta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)


MODEL_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"
DEVICE = torch.device("cuda:1")
EMBED_SIZE = 256
PASSAGE_BATCH_SIZE = 100000

net = network.DSSM(embed_size=EMBED_SIZE)
net.load_state_dict(torch.load(MODEL_PATH))
net.to(DEVICE)
net.eval()

######################
# True embeddings
print_message("Loading passage embeddings")
EMBEDDING_PATH = "/home/jianx/data/results/passage_embeddings.dict"
passage_embeddings = obj_reader(EMBEDDING_PATH)
print_message("Embeddings successfully loaded")

#############################
# Test embeddings
# print_message("Generating fake embeddings")
# passage_embeddings = {}
# for i in range(8800000):
#     passage_embeddings[i * 100] = list(range(256))
# print_message("Embeddings successfully generated")

query_test_dict = obj_reader("/home/jianx/data/queries_test.dict")
rating_dict = obj_reader("/home/jianx/data/rel_scores.dict")

qids = list(rating_dict.keys())
query_embeddings = []
for qid in qids:
    q_seq = query_test_dict[qid]
    query_embeddings.append(net(generate_sparse(q_seq).to(DEVICE)).detach())
query_embedding_tensor = torch.stack(query_embeddings, dim=0).unsqueeze(dim=1).to(DEVICE)

pids = list(passage_embeddings.keys())
pids_2d = []
slice_idx = 0
while slice_idx + PASSAGE_BATCH_SIZE < len(pids):
    pid_batch = pids[slice_idx:(slice_idx + PASSAGE_BATCH_SIZE)]
    slice_idx += PASSAGE_BATCH_SIZE
    pids_2d.append(pid_batch)
pids_2d.append(pids[slice_idx:])
cosine_similarities_tensor = torch.empty([len(qids), 0]).to(DEVICE)
for i, batch in enumerate(pids_2d):
    print_message("Calculating Cosine Similarities: Batch No." + str(i+1) + "/" + str(len(pids_2d)))
    cosine_sim_calc = torch.nn.CosineSimilarity(dim=-1)
    passage_batch_embeddings = []
    for pid in batch:
        passage_batch_embeddings.append(torch.FloatTensor(passage_embeddings[pid]))
    passage_batch_embedding_tensor = torch.stack(passage_batch_embeddings, dim=0).unsqueeze(dim=0).to(DEVICE)
    sim = cosine_sim_calc(query_embedding_tensor, passage_batch_embedding_tensor)
    cosine_similarities_tensor = torch.cat([cosine_similarities_tensor, sim], dim=1)

print_message("Finished calculating cosine similarities.")
result_dict = {}
for i, qid in enumerate(qids):
    print_message("Processing query: " + str(qid) + " No." + str(i+1) + "/" + str(len(qids)))
    scores = cosine_similarities_tensor[i].cpu()
    q_results = dict({})
    for j, pid in enumerate(pids):
        q_results[pid] = scores[j].item()
    q_results = dict(sorted(q_results.items(), key=lambda x: x[1], reverse=True)[:1000])
    result_dict[qid] = q_results

print_message("Calculating metrics.")
result_ndcg = []
result_prec = []
result_rr = []
for qid in qids:
    if len(result_dict[qid]) < 10:
        print_message(qid)
    ndcg, prec, rr = get_ndcg_precision_rr(rating_dict[qid], result_dict[qid], 10)
    result_ndcg.append(ndcg)
    result_prec.append(prec)
    result_rr.append(rr)
print_message(result_ndcg)
avg_ndcg = np.nanmean(result_ndcg)
avg_prec = np.nanmean(result_prec)
avg_rr = np.nanmean(result_rr)
print_message("NDCG: " + str(avg_ndcg))
print_message("Precision: " + str(avg_prec))
print_message("RR: " + str(avg_rr))
print_message("Saving result dictionary.")
obj_writer(result_dict, "/home/jianx/data/results/brute_search_result.dict")
print_message("Successfully saved.")