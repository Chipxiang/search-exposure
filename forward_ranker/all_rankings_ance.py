import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

IS_FLAT = False
BATCH_SIZE = 100
NLIST = 100
RANK = 100
OUT_PATH = "/datadrive/jianx/data/results/all_search_rankings_{}_{}_{}.csv".format(NLIST, RANK,
                                                                                   "flat" if IS_FLAT else "approximate")

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")
query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")
query_train_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_query_train_map.dict")
pid_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_passage_map.dict")

print_message("Building index")
faiss.omp_set_num_threads(16)
dim = passage_embeddings.shape[1]
if IS_FLAT:
    cpu_index = faiss.IndexFlatIP(dim)
else:
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, NLIST)
    assert not cpu_index.is_trained
    cpu_index.train(passage_embeddings)
    assert cpu_index.is_trained

cpu_index.add(passage_embeddings)
print_message("Searching for all queries")
with open(OUT_PATH, "w+") as f:
    print_message("Writing to {}".format(OUT_PATH))
    f.write("")
for starting in range(0, len(query_train_embeddings), BATCH_SIZE):
    mini_batch = query_train_embeddings[starting:starting + BATCH_SIZE]
    _, dev_I = cpu_index.search(mini_batch, RANK)
    print_message("Batch No.{}/{}".format(starting / BATCH_SIZE + 1, len(query_train_embeddings) / BATCH_SIZE))
    for idx, top_list in enumerate(dev_I):
        qid = query_train_mapping[idx + starting]
        with open(OUT_PATH, "a") as f:
            for j, index in enumerate(top_list):
                f.write("{},{},{}\n".format(qid, pid_mapping[index], j + 1))
