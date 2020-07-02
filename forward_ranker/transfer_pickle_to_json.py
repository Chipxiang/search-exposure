from load_data import obj_writer
import pickle

with open("/home/jianx/data/results/passage_embeddings.dict", 'rb') as handle:
    dictionary = pickle.load(handle)
obj_writer(dictionary, "/home/jianx/data/results/passage_embeddings.dict")
