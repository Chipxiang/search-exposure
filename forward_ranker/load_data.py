import pickle

POS_NEG_DICT_PATH = "/home/jianx/data/qid_pos_neg.dict"
PASSAGE_DICT_PATH = "/home/jianx/data/passages.dict"
QUERY_TRAIN_DICT_PATH = "/home/jianx/data/queries_train.dict"
QUERY_EVAL_DICT_PATH = "/home/jianx/data/queries_eval.dict"
QUERY_DEV_DICT_PATH = "/home/jianx/data/queries_dev.dict"
TOP_DICT_PATH = "/home/jianx/data/initial_ranking.dict"
RATING_DICT_PATH = "/home/jianx/data/rel_scores.dict"
QUERY_TEST_DICT_PATH = "/home/jianx/data/queries_test.dict"


def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")


def obj_writer(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load():
    pos_neg_dict = obj_reader(POS_NEG_DICT_PATH)
    query_dict = obj_reader(QUERY_TRAIN_DICT_PATH)
    passage_dict = obj_reader(PASSAGE_DICT_PATH)
    top_dict = obj_reader(TOP_DICT_PATH)
    rating_dict = obj_reader(RATING_DICT_PATH)
    query_test_dict = obj_reader(QUERY_TEST_DICT_PATH)
    return pos_neg_dict, query_dict, passage_dict, top_dict, rating_dict, query_test_dict
