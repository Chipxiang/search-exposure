import pickle

POS_NEG_DICT_PATH = "/home/jianx/data/qid_pos_neg.dict"
PASSAGE_DICT_PATH = "/home/jianx/data/passages.dict"
QUERY_TRAIN_DICT_PATH = "/home/jianx/data/queries_train.dict"
QUERY_EVAL_DICT_PATH = "/home/jianx/data/queries_eval.dict"
QUERY_DEV_DICT_PATH = "/home/jianx/data/queries_dev.dict"


def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.loads(handle.read())


def load():
    pos_neg_dict = obj_reader(POS_NEG_DICT_PATH)
    query_dict = obj_reader(QUERY_TRAIN_DICT_PATH)
    passage_dict = obj_reader(PASSAGE_DICT_PATH)
    return pos_neg_dict, query_dict, passage_dict

