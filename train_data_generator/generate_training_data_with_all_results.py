import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import numpy as np
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

BATCH_SIZE = 10000
SAMPLE_SIZE = 8841823
RANK = 100
TRAINING_DATA_PATH = "/datadrive/jianx/data/train_data/ance_training_rank{}_{}.csv".format(RANK, SAMPLE_SIZE)

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")

query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")
query_train_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_query_train_map.dict")
pid_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_passage_map.dict")
pid_offset = obj_reader("/datadrive/data/preprocessed_data_with_test/pid2offset.pickle")

print_message("Loading full search results")
all_results = {}
with open("/datadrive/jianx/data/results/all_search_rankings_100_100_flat.csv", "r") as f:
    for line in f:
        qid = int(line.split(",")[0])
        pid = int(line.split(",")[1])
        rank = int(line.split(",")[2])
        if qid not in all_results.keys():
            all_results[qid] = {}
        all_results[qid][pid] = rank

print_message("Building index")
faiss.omp_set_num_threads(16)
dim = passage_embeddings.shape[1]
query_index = faiss.IndexFlatIP(dim)
query_index.add(query_train_embeddings)

with open(TRAINING_DATA_PATH, "w+") as f:
    f.write("")
avg_match = 0

for starting in range(0, SAMPLE_SIZE, BATCH_SIZE):

    mini_batch = passage_embeddings[starting:starting + BATCH_SIZE]
    _, queries_idx = query_index.search(mini_batch, RANK)
    for passage_idx, indices in enumerate(queries_idx):
        pid = pid_mapping[passage_idx + starting]
        with open(TRAINING_DATA_PATH, "a") as f:
            for q_idx in indices:
                qid = query_train_mapping[q_idx]
                if pid in all_results[qid].keys():
                    avg_match += 1
                f.write("{},{},{}\n".format(pid, qid, all_results[qid].get(pid, 0)))

    print_message("Processed passage No. {}/{}, Avg match: {}".format(starting + BATCH_SIZE, SAMPLE_SIZE,
                                                                      avg_match / (starting + BATCH_SIZE)))
print_message("Finished generating training data.")
