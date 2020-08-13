import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

PASSAGE_DICT_PATH = "/datadrive/jianx/data/passages.dict"
QUERY_TRAIN_DICT_PATH = "/datadrive/jianx/data/queries_train.dict"

# Queries plain text: queries.train.tsv
# Passages plain text: collection.tsv
QUERIES_TEXT_PATH = "/datadrive/jianx/data/queries.train.tsv"
PASSAGES_TEXT_PATH = "/datadrive/jianx/data/collection.tsv"

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")
query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")

from sklearn.decomposition import PCA
all_embeddings = np.concatenate((passage_embeddings, query_train_embeddings), axis = 0)
print(all_embeddings.shape)
pca = PCA(n_components=50)
pca.fit(all_embeddings)

print("PCA Explained Variance: {}%".format(np.round(sum(pca.explained_variance_ratio_)*100, 4)))
all_embeddings_pca = pca.transform(all_embeddings)

from sklearn.manifold import TSNE
all_embeddings_tsne = TSNE(n_components=2).fit_transform(all_embeddings_pca)
passage_path = "/datadrive/ruohan/tsne/passage_tsne.pb"
query_path = "/datadrive/ruohan/tsne/query_tsne.pb"
passage_tsne = all_embeddings_tsne[:passage_embeddings.shape[0],:]
query_tsne = all_embeddings_tsne[passage_embeddings.shape[0]:,:]
obj_writer(passage_tsne, passage_path)
obj_writer(query_tsne, query_path)
