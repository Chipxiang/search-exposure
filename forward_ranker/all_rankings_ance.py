import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import faiss
import forward_ranker.load_data as load_data
from forward_ranker.utils import print_message

obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer

RANK = 100
OUT_PATH = "/datadrive/results/all_search_rankings_{}.csv".format(RANK)

print_message("Loading embeddings.")
passage_embeddings = obj_reader("/home/jianx/results/passage_0__emb_p__data_obj_0.pb")
query_train_embeddings = obj_reader("/home/jianx/results/query_0__emb_p__data_obj_0.pb")
query_train_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_query_train_map.dict")
pid_mapping = obj_reader("/datadrive/jianx/data/annoy/100_ance_passage_map.dict")

print_message("Building index")
faiss.omp_set_num_threads(16)
dim = passage_embeddings.shape[1]
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embeddings)
print_message("Searching for all queries")

_, dev_I = cpu_index.search(query_train_embeddings, RANK)
print_message("Writing results")
for idx, top_list in enumerate(dev_I):
    qid = query_train_mapping[idx]
    with open(OUT_PATH) as f:
        for j, index in enumerate(top_list):
            f.write("{},{},{}\n".format(qid, pid_mapping[index], j + 1))
    if idx % 100_000 == 0:
        print_message("Written {}/{}".format(idx, len(dev_I)))
