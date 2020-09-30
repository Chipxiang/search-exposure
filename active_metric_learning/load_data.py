import csv
import numpy as np
from util import obj_reader

PASSAGE_NP_PATH = "/home/jianx/results/passage_0__emb_p__data_obj_0.pb"
PASSAGE_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_passage_map.dict"
QUERY_TRAIN_NP_PATH = "/home/jianx/results/query_0__emb_p__data_obj_0.pb"
QUERY_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_query_train_map.dict"


OUT_RANK = 200

def load_train(path):
    with open(path, "r") as file:
        pos_dict = {}
        neg_dict = {}
        count = 0
        for line in file:
            tokens = line.split(",")
            pid = int(tokens[0])
            qid = int(tokens[1])
            rank = int(tokens[2].rstrip())
            if rank == 0:
                if pid not in neg_dict:
                    neg_dict[pid] = {}
                neg_dict[pid][qid] = OUT_RANK
            else:
                if pid not in pos_dict:
                    pos_dict[pid] = {}
                pos_dict[pid][qid] = rank
    return pos_dict, neg_dict
def map_id(old_np, mapping):
    new_dict = dict(zip(mapping.values(),old_np))
    return new_dict
def load(train_rank_path):
    print("Load embeddings.")
    passage_np = obj_reader(PASSAGE_NP_PATH)
    pid_mapping = obj_reader(PASSAGE_MAP_PATH)
    query_np = obj_reader(QUERY_TRAIN_NP_PATH)
    qid_mapping = obj_reader(QUERY_MAP_PATH)
    print("Mapping ids.")
    query_dict = map_id(query_np, qid_mapping)
    passage_dict = map_id(passage_np, pid_mapping)
    print("Load training data.")
    train_pos_dict, train_neg_dict = load_train(train_rank_path)
    return train_pos_dict, train_neg_dict, query_dict, passage_dict