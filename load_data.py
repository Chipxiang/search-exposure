import pickle

POSITIVE_DICT_PATH = "/home/jianx/data/positive_qid_pid.dict"
NEGATIVE_DICT_PATH = "/home/jianx/data/negative_qid_pid.dict"
PASSAGE_DICT_PATH = "/home/jianx/data/passages.dict"
QUERY_TRAIN_DICT_PATH = "/home/jianx/data/queries_train.dict"
QUERY_EVAL_DICT_PATH = "/home/jianx/data/queries_eval.dict"
QUERY_DEV_DICT_PATH = "/home/jianx/data/queries_dev.dict"


def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.loads(handle.read())


def load():
    positive_dict = obj_reader(POSITIVE_DICT_PATH)
    negative_dict = obj_reader(NEGATIVE_DICT_PATH)
    query_dict = obj_reader(QUERY_TRAIN_DICT_PATH)
    passage_dict = obj_reader(PASSAGE_DICT_PATH)
    return positive_dict, negative_dict, query_dict, passage_dict

