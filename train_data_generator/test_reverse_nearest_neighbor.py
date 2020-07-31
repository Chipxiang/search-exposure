import sys

sys.path.insert(0, '/home/jianx/search-exposure/forward_ranker/')
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import generate_sparse
from load_data import obj_reader, obj_writer
import network
from annoy import AnnoyIndex
from utils import print_message

# (Assume annoy is working)
# Test on 1000 random documents
# Start with a document D, find nearest N queries from a query (train) annoy index.
# For each query from N, get nearest 1000 documents.(Annoy) (Ground Truth)
# Calculate average ranking of document D in these queries. (Plot)
# Count how many of the queries in N ranks D in top 10

EMBED_SIZE = 256
NUM_OF_DOCUMENTS = 10000
NUM_OF_NEAREST_QUERIES = 1000
TOP_K_RANKING = 100

MODEL_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"
DEVICE = torch.device("cuda")
NET = network.DSSM(embed_size=EMBED_SIZE)
NET.load_state_dict(torch.load(MODEL_PATH))
NET.to(DEVICE)
NET.eval()

print_message("Loading annoy indices.")
passage_index = AnnoyIndex(EMBED_SIZE, 'euclidean')
passage_index.load("/home/jianx/data/annoy/128_passage_index.ann")
pid_mapping = obj_reader("/home/jianx/data/annoy/128_pid_map.dict")

query_index = AnnoyIndex(EMBED_SIZE, 'euclidean')
query_index.load("/home/jianx/data/annoy/128_query_index.ann")
qid_mapping = obj_reader("/home/jianx/data/annoy/128_qid_map.dict")

print_message("Loading passage embeddings.")
passage_embeddings = obj_reader("/home/jianx/data/results/passage_embeddings.dict")
print_message("Sampling " + str(NUM_OF_DOCUMENTS) + " documents.")
passage_embeddings = dict(random.sample(list(passage_embeddings.items()), NUM_OF_DOCUMENTS))
print_message("Loading query dictionary.")
query_train_dict = obj_reader("/home/jianx/data/queries_train.dict")

counter = 0
rankings = []
for pid, embedding in passage_embeddings.items():
    print_message("Processing passage No. " + str(counter) + "/" + str(NUM_OF_DOCUMENTS))
    nearest_queries = query_index.get_nns_by_vector(embedding, NUM_OF_NEAREST_QUERIES)
    matching_list = []

    for i, annoy_qid in enumerate(nearest_queries):
        qid = qid_mapping[annoy_qid]
        top_list = passage_index.get_nns_by_vector(NET(generate_sparse(query_train_dict[qid]).to(DEVICE)).detach(),
                                                   TOP_K_RANKING)
        is_matched = False
        for j, annoy_pid in enumerate(top_list):
            if pid_mapping[annoy_pid] == pid:
                matching_list.append(j + 1)
                is_matched = True
                break
        if not is_matched:
            matching_list.append(0)
    rankings.append(matching_list)
    counter += 1

rankings_array = np.array(rankings)
obj_writer(rankings_array, "/home/jianx/data/train_data/test_rankings_10000.np")

print_message("Avg No. of matching: " + str(np.count_nonzero(rankings_array) / len(rankings_array)))
mean_rank = 0
for matching_array in rankings_array:
    if np.count_nonzero(matching_array) != 0:
        mean_rank += np.sum(matching_array) / np.count_nonzero(matching_array)
mean_rank /= len(rankings_array)
print_message("Avg rank of each document: " + str(mean_rank))
