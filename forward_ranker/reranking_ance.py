import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message
import numpy as np

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

IS_FLAT = True
BATCH_SIZE = 100
NLIST = 100
RANK = 100
OUT_PATH = "/datadrive/jianx/data/results/rerank_search_rankings_{}_{}_{}.csv".format(NLIST, RANK,
                                                                                   "flat" if IS_FLAT else "approximate")
print_message("Loading top 1000s")
top_1000 = obj_reader("/datadrive/jianx/data/top1000_train.dict")
# top_1000 = {}
#
# with open("/datadrive/ruohan/data/top1000.train.txt", "r") as f:
#     for line in f:
#         split = line.split("\t")
#         if int(split[0]) not in top_1000:
#             top_1000[int(split[0])] = []
#         top_1000[int(split[0])].append(int(split[1]))

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")
query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")
query_train_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_query_train_map.dict")
pid_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_passage_map.dict")
pid_offset = obj_reader("/datadrive/data/preprocessed_data_with_test/pid2offset.pickle")



# print_message("Building index")
# faiss.omp_set_num_threads(16)
# dim = passage_embeddings.shape[1]
# if IS_FLAT:
#     cpu_index = faiss.IndexFlatIP(dim)
# else:
#     quantizer = faiss.IndexFlatIP(dim)
#     cpu_index = faiss.IndexIVFFlat(quantizer, dim, NLIST)
#     assert not cpu_index.is_trained
#     cpu_index.train(passage_embeddings)
#     assert cpu_index.is_trained
#
# cpu_index.add(passage_embeddings)

faiss.omp_set_num_threads(16)
dim = passage_embeddings.shape[1]

print_message("Searching for all queries")
with open(OUT_PATH, "w+") as f:
    print_message("Writing to {}".format(OUT_PATH))
    f.write("")

for idx, q_emb in enumerate(query_train_embeddings):
    qid = query_train_mapping[idx]
    index = faiss.IndexFlatIP(dim)
    p_indices = []
    if qid in top_1000:
        for i in top_1000[qid]:
            if i in pid_offset:
                p_indices.append(pid_offset[i])
    if len(p_indices) == 0:
        continue
    p_embs = passage_embeddings[np.array(p_indices)]
    index.add(p_embs)
    _, top_list = index.search(np.array([q_emb]), RANK)
    with open(OUT_PATH, "a") as f:
        for j, k in enumerate(top_list[0]):
            f.write("{},{},{}\n".format(qid, top_1000[qid][k], j + 1))
    if idx % 10000 == 0:
        print_message("Processing {}/{}".format(idx, len(query_train_embeddings)))
