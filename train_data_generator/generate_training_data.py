import sys
sys.path.insert(0, '/home/jianx/search-exposure/forward_ranker/')
import random
import torch
from train import generate_sparse
from load_data import obj_reader, obj_writer
import network
from annoy import AnnoyIndex
from utils import print_message

EMBED_SIZE = 256
NUM_OF_DOCUMENTS = 200_000
NUM_OF_NEAREST_QUERIES = 100
TOP_K_RANKING = 100
TRAINING_DATA_PATH = "/home/jianx/data/train_data/{}_{}_{}_{}_training.csv".format(EMBED_SIZE, NUM_OF_DOCUMENTS,
                                                                                   NUM_OF_NEAREST_QUERIES,
                                                                                   TOP_K_RANKING)
with open(TRAINING_DATA_PATH, "w+") as f:
    f.write("")

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
for pid, embedding in passage_embeddings.items():
    if counter % 500 == 0:
        print_message("Processing passage No. " + str(counter + 1) + "/" + str(NUM_OF_DOCUMENTS))
    nearest_queries = query_index.get_nns_by_vector(embedding, NUM_OF_NEAREST_QUERIES)

    for i, annoy_qid in enumerate(nearest_queries):
        qid = qid_mapping[annoy_qid]
        top_list = passage_index.get_nns_by_vector(NET(generate_sparse(query_train_dict[qid]).to(DEVICE)).detach(),
                                                   TOP_K_RANKING)
        for j, annoy_pid in enumerate(top_list):
            if pid_mapping[annoy_pid] == pid:
                with open(TRAINING_DATA_PATH, "a") as f:
                    f.write("{},{},{}\n".format(pid, qid, j + 1))
                break
    counter += 1
