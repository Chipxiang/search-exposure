from rank_bm25 import BM25L
from load_data import obj_reader
import numpy as np

COLLECTION_PATH = "/home/jianx/data/collection.tsv"
OUTPUT_PATH = "/home/jianx/data/bm25/L.qrels"
corpus = []
with open(COLLECTION_PATH, "r") as f:
    line = f.readline()
    while line:
        corpus.append(line.split("\t")[1])
        line = f.readline()
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25L(tokenized_corpus)
initial_ranking = obj_reader("/home/jianx/data/initial_ranking.dict")
QUERY_TEST_PATH = "/home/jianx/data/msmarco-test2019-queries.tsv"
query_test = {}
with open(QUERY_TEST_PATH, "r") as f:
    line = f.readline()
    while line:
        query_test[int(line.split("\t")[0])] = line.split("\t")[1]
        line = f.readline()
print(len(query_test))

rating_dict = obj_reader("/home/jianx/data/rel_scores.dict")
print(rating_dict.keys())

result_dict = {}
for qid, query in query_test.items():
    if qid not in rating_dict.keys():
        continue
    print("Processing:" + str(qid))
    scores = np.array(bm25.get_scores(query.split(" ")))
    result_list = np.argsort(scores)[::-1][:1000]
    result_scores = np.sort(scores)[::-1][:1000]
    with open(OUTPUT_PATH, "a+") as f:
        rank = 1
        for i, docid in enumerate(result_list):
            f.write("{} {} {} {} {} {}\n".format(qid, "Q0", docid, rank, result_scores[i], "STANDARD"))
            rank += 1