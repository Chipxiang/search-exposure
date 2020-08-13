import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import numpy as np
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

SAMPLE_SIZE = 1000
RANK = 100
TRAINING_DATA_PATH = "/datadrive/jianx/data/train_data/ance_training_rank{}_{}.csv".format(RANK, SAMPLE_SIZE)

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")
query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")
query_train_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_query_train_map.dict")
pid_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_passage_map.dict")
pid_offset = obj_reader("/datadrive/data/preprocessed_data_with_test/pid2offset.pickle")

print_message("Building index")
faiss.omp_set_num_threads(16)
dim = passage_embeddings.shape[1]
passage_index = faiss.IndexFlatIP(dim)
passage_index.add(passage_embeddings)
query_index = faiss.IndexFlatIP(dim)
query_index.add(query_train_embeddings)

print_message("Searching all passages")
_, queries_idx = query_index.search(passage_embeddings[:SAMPLE_SIZE], RANK)

with open(TRAINING_DATA_PATH, "w+") as f:
    f.write("")

avg_match = 0
for passage_idx, indices in enumerate(queries_idx):
    nearest_q_embs = []
    for idx in indices:
        nearest_q_embs.append(query_train_embeddings[idx])
    nearest_q_embs_array = np.array(nearest_q_embs)
    _, passages = passage_index.search(nearest_q_embs_array, RANK)
    match_idx = np.where(passages == passage_idx)
    avg_match += len(match_idx[0])
    with open(TRAINING_DATA_PATH, "a") as f:
        for i, query_results in enumerate(passages):
            match = np.where(query_results == passage_idx)
            rank = 0
            if len(match[0]) != 0:
                rank = match[0][0] + 1
            f.write("{},{},{}\n".format(pid_mapping[passage_idx], query_train_mapping[i], rank))
    print_message("Processed passage No. {}/{}, Avg match: {}".format(passage_idx + 1, SAMPLE_SIZE,
                                                                      avg_match / (passage_idx + 1)))

print_message("Average number of exposure quereis found: {}".format(avg_match / SAMPLE_SIZE))
