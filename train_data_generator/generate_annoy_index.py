import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import torch
from annoy import AnnoyIndex
import forward_ranker.network as network
import forward_ranker.load_data as load_data
import forward_ranker.train as train

generate_sparse = train.generate_sparse
obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

MODEL_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"
DEVICE = torch.device("cuda")
EMBED_SIZE = 256
TREE_SIZE = 1024


def generate_annoy_index(embed_size, embeddings):
    mapping = {}
    i = 0
    index = AnnoyIndex(embed_size, 'euclidean')
    for key, value in embeddings.items():
        index.add_item(i, value)
        mapping[key] = i
        i += 1
        if i % 10000 == 0:
            print("Progress: " + str(i) + "/" + str(len(embeddings)) + " " + str(i / len(embeddings)))
    return index, mapping


PASSAGE_EMBEDDINGS = obj_reader("/home/jianx/data/results/passage_embeddings.dict")

PID_INDEX, PID_MAP = generate_annoy_index(EMBED_SIZE, PASSAGE_EMBEDDINGS)
del PASSAGE_EMBEDDINGS
PID_INDEX.build(TREE_SIZE)
PID_INDEX.save("/home/jianx/data/annoy/" + str(TREE_SIZE) + "_passage_index.ann")
obj_writer(PID_MAP, "/home/jianx/data/annoy/" + str(TREE_SIZE) + "_pid_map.dict")
