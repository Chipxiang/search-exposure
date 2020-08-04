import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import torch
from annoy import AnnoyIndex
import forward_ranker.load_data as load_data
import forward_ranker.train as train
from forward_ranker.utils import print_message
import pickle
import numpy as np

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

EMBEDDING_PATH = sys.argv[1]
RAW_PATH = sys.argv[2]
OUT_DIR = sys.argv[3]
TREE_SIZE = int(sys.argv[4])


def generate_annoy_index(embed_size, embeddings, raw_data_path):
    embedding_dict = {}
    with open(raw_data_path, "r") as f:
        rc = 0
        for line in f:
            embedding_dict[int(line.split("\t")[0])] = embeddings[rc]
            rc += 1
    mapping = {}
    i = 0
    index = AnnoyIndex(embed_size, 'dot')
    for key, value in embedding_dict.items():
        index.add_item(i, value)
        mapping[i] = key
        i += 1
        if i % 1_000_000 == 0:
            print_message("Progress: " + str(i) + "/" + str(len(embedding_dict)) + " " + str(i / len(embeddings)))
    return index, mapping


passage_embeddings = obj_reader(EMBEDDING_PATH)
print(passage_embeddings.shape)

passage_index, passage_mapping = generate_annoy_index(768, passage_embeddings,
                                                      RAW_PATH)
del passage_embeddings
obj_writer(passage_mapping, OUT_DIR + str(TREE_SIZE) + "_ance_passage_map.dict")
del passage_mapping

print_message("Start Building.")
passage_index.build(TREE_SIZE)
print_message("Finished Building.")

passage_index.save(OUT_DIR + str(TREE_SIZE) + "_ance_passage_index.ann")
print_message("Successfully Saved.")
