import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import random
import numpy as np
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message
from forward_ranker.utils import timestamp

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

N_QUERIES = 50_000
TRAIN_SIZE = 200_000
TEST_SIZE = 20_000
BATCH_SIZE = 20_000
RANK = 100

TRAINING_DATA_PATH = "/datadrive/jianx/data/train_data/ance_training_rank{}_nqueries{}_{}_{}.csv".format(
    RANK,
    N_QUERIES,
    TRAIN_SIZE,
    timestamp())
TEST_DATA_PATH = "/datadrive/jianx/data/train_data/ance_testing_rank{}_nqueries{}_{}_{}.csv".format(
    RANK,
    N_QUERIES,
    TEST_SIZE,
    timestamp())
all_lcs_set = set(random.sample(range(8841823), TRAIN_SIZE + TEST_SIZE))
test_lcs_set = set(random.sample(all_lcs_set, TEST_SIZE))
train_lcs_set = all_lcs_set - test_lcs_set
train_lcs = np.array(list(train_lcs_set))
test_lcs = np.array(list(test_lcs_set))

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
query_index.add(query_train_embeddings[:N_QUERIES])
full_query_index = faiss.IndexFlatIP(dim)
full_query_index.add(query_train_embeddings)

train_passage_embeddings = passage_embeddings[train_lcs]
test_passage_embeddings = passage_embeddings[test_lcs]
with open(TRAINING_DATA_PATH, "w+") as f:
    f.write("")
with open(TEST_DATA_PATH, "w+") as f:
    f.write("")

avg_match = 0

# Training
print_message("Start training.")
for starting in range(0, TRAIN_SIZE, BATCH_SIZE):
    mini_batch = train_passage_embeddings[starting:starting + BATCH_SIZE]
    _, queries_idx = query_index.search(mini_batch, RANK)
    for passage_idx, indices in enumerate(queries_idx):
        pid = pid_mapping[train_lcs[passage_idx + starting]]
        with open(TRAINING_DATA_PATH, "a") as f:
            for q_idx in indices:
                qid = query_train_mapping[q_idx]
                if pid in all_results[qid].keys():
                    avg_match += 1
                f.write("{},{},{}\n".format(pid, qid, all_results[qid].get(pid, 0)))

    print_message("Processed passage No. {}/{}, Avg match: {}".format(starting + BATCH_SIZE, TRAIN_SIZE,
                                                                      avg_match / (starting + BATCH_SIZE)))
# Testing
avg_match = 0
print_message("Start testing.")
for starting in range(0, TEST_SIZE, BATCH_SIZE):
    mini_batch = test_passage_embeddings[starting:starting + BATCH_SIZE]
    _, queries_idx = full_query_index.search(mini_batch, RANK)
    for passage_idx, indices in enumerate(queries_idx):
        pid = pid_mapping[test_lcs[passage_idx + starting]]
        with open(TEST_DATA_PATH, "a") as f:
            for q_idx in indices:
                qid = query_train_mapping[q_idx]
                if pid in all_results[qid].keys():
                    avg_match += 1
                f.write("{},{},{}\n".format(pid, qid, all_results[qid].get(pid, 0)))

    print_message("Processed passage No. {}/{}, Avg match: {}".format(starting + BATCH_SIZE, TEST_SIZE,
                                                                      avg_match / (starting + BATCH_SIZE)))
print_message("Finished generating data.")
