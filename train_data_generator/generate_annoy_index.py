import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import torch
from annoy import AnnoyIndex
import forward_ranker.load_data as load_data
import forward_ranker.train as train
import datetime
from datetime import datetime, timezone, timedelta

generate_sparse = train.generate_sparse
obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

MODEL_PATH = "/home/jianx/data/results/100_1000_1000_0.001_256_10.model"
DEVICE = torch.device("cuda")
EMBED_SIZE = 256
TREE_SIZE = 128

TIME_OFFSET = -4


def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedelta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)


def generate_annoy_index(embed_size, embeddings):
    mapping = {}
    i = 0
    index = AnnoyIndex(embed_size, 'euclidean')
    for key, value in embeddings.items():
        index.add_item(i, value)
        mapping[i] = key
        i += 1
        if i % 50000 == 0:
            print_message("Progress: " + str(i) + "/" + str(len(embeddings)) + " " + str(i / len(embeddings)))
    return index, mapping


print_message("Start Loading Embeddings")
PASSAGE_EMBEDDINGS = obj_reader("/home/jianx/data/results/passage_embeddings.dict")
print_message("Embeddings Successfully Loaded")

PID_INDEX, PID_MAP = generate_annoy_index(EMBED_SIZE, PASSAGE_EMBEDDINGS)
del PASSAGE_EMBEDDINGS
print_message("Start Building.")
PID_INDEX.build(TREE_SIZE)
print_message("Finished Building.")

PID_INDEX.save("/home/jianx/data/annoy/" + str(TREE_SIZE) + "_passage_index.ann")
obj_writer(PID_MAP, "/home/jianx/data/annoy/" + str(TREE_SIZE) + "_pid_map.dict")
print_message("Successfully Saved.")
