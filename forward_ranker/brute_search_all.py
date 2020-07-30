import torch
from train import generate_sparse
from load_data import obj_reader
from load_data import obj_writer
import network
import numpy as np
import datetime
from datetime import datetime, timezone, timedelta

TIME_OFFSET = -4


def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedlta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)


MODEL_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"
DEVICE = torch.device("cuda:2")
EMBED_SIZE = 256
PASSAGE_BATCH_SIZE = 8

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

############################
# Test embeddings
# print_message("Generating fake embeddings")
# passage_embeddings = {}
# for i in range(8800000):
#     passage_embeddings[i * 100] = np.random.rand(256).tolist()
# print_message("Embeddings successfully generated")

query_train_dict = obj_reader("/home/jianx/data/queries_train.dict")
qids = []
print_message("Generating query embeddings.")
query_embeddings = []
counter = 0

# True Embeddings
for qid, q_seq in query_train_dict.items():
    if len(q_seq) != 0:
        query_embeddings.append(net(generate_sparse(q_seq).to(DEVICE)).detach())
        qids.append(qid)
    counter += 1
    if (counter - 1) % 10000 == 0:
        print_message("Generated query embeddings: {}/{}".format(counter, len(query_train_dict)))
query_embedding_tensor = torch.stack(query_embeddings, dim=0).unsqueeze(dim=1).to(DEVICE)

# Test Embeddings
# for qid, q_seq in query_train_dict.items():
#     query_embeddings.append(torch.FloatTensor(np.random.rand(256)))
# query_embedding_tensor = torch.stack(query_embeddings, dim=0).unsqueeze(dim=1).to(DEVICE)

print_message("Dividing passage embedding into batches.")
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
    print_message("Calculating Cosine Similarities: Batch No." + str(i + 1) + "/" + str(len(pids_2d)))
    cosine_sim_calc = torch.nn.CosineSimilarity(dim=-1)
    passage_batch_embeddings = []
    for pid in batch:
        passage_batch_embeddings.append(torch.FloatTensor(passage_embeddings[pid]))
    passage_batch_embedding_tensor = torch.stack(passage_batch_embeddings, dim=0).unsqueeze(dim=0).to(DEVICE)
    sim = cosine_sim_calc(query_embedding_tensor, passage_batch_embedding_tensor)
    cosine_similarities_tensor = torch.cat([cosine_similarities_tensor, sim], dim=1)

print_message("Finished calculating cosine similarities.")
result_dict = {}
pids_tensor = torch.tensor(pids)
for i, qid in enumerate(qids):
    print_message("Processing query: " + str(qid) + " No." + str(i + 1) + "/" + str(len(qids)))
    scores = cosine_similarities_tensor[i].cpu()
    sorted_score = scores.sort(descending=True)[:1000]
    score_ids = scores.argsort(descending=True)[:1000]
    q_results = dict(zip(pids_tensor[score_ids].tolist(), sorted_score.tolist()))
    print_message("Query ")
    result_dict[qid] = q_results

print_message("Saving result dictionary.")
obj_writer(result_dict, "/datadrive/brute_search_all_result.dict")
print_message("Successfully saved.")
