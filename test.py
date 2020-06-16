import torch
import random

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
            if p_seq == []:
                score = -1
            else:
                p_embed = net(generate_sparse(p_seq).to(device))
                score = torch.cosine_similarity(q_embed.unsqueeze(0), p_embed.unsqueeze(0)).item()
            q_results[pid] = score
        result_dict[qid] = q_results
    return result_dict