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
TREE_SIZE = 128

NET = network.DSSM(embed_size=EMBED_SIZE)
NET.load_state_dict(torch.load(MODEL_PATH))
NET.to(DEVICE)
NET.eval()

QUERY_TRAIN_DICT_PATH = "/home/jianx/data/queries_train.dict"
QUERY_DICT = obj_reader(QUERY_TRAIN_DICT_PATH)


def generate_annoy_index(net, device, embed_size, dictionary):
    mapping = {}
    i = 0
    index = AnnoyIndex(embed_size, 'euclidean')
    for key, value in dictionary.items():
        if len(value) != 0:
            index.add_item(i, net(generate_sparse(value).to(device)).detach().tolist())
            mapping[i] = key
            i += 1
        if i % 10000 == 0:
            print("Progress: " + str(i) + "/" + str(len(dictionary)) + " " + str(i / len(dictionary)))
    return index, mapping


QID_INDEX, QID_MAP = generate_annoy_index(NET, DEVICE, EMBED_SIZE, QUERY_DICT)
QID_INDEX.build(TREE_SIZE)
QID_INDEX.save("/home/jianx/data/annoy/" + str(TREE_SIZE) + "_query_index.ann")
obj_writer(QID_MAP, "/home/jianx/data/annoy/" + str(TREE_SIZE) + "_qid_map.dict")
