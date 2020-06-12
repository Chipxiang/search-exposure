import csv

PASSAGE_PATH = "/home/jianx/data/passage_indices.csv"
QUERY_TRAIN_PATH = "/home/jianx/data/queries.train.tsv"
TRAIN_TRIPLE_PATH = "/home/jianx/data/qidpidtriples.train.full.2.tsv"

def id_content_reader(path):
    reader = csv.reader(open(path))
    result = {}
    for row in reader:
        result[int(row[0])] = list(map(int, row[1].split()))
    return result

def pos_neg_dict_reader(path):
    file = open(path)
    line = file.readline()
    pos_dict = {}
    neg_dict = {}
    while line:
        ids = list(map(int, line.split()))
        pos_dict[ids[0]] = ids[1]
        neg_dict[ids[0]] = [ids[2]]
        current_id = ids[0]
        line = file.readline()
        ids = list(map(int, line.split()))
        while line and current_id == ids[0]:
            neg_dict[current_id].append(int(ids[2]))
            line = file.readline()
            ids = list(map(int, line.split()))
    return pos_dict,neg_dict

def load():
	positive_dict, negative_dict = pos_neg_dict_reader(TRAIN_TRIPLE_PATH)
	query_dict = id_content_reader(QUERY_TRAIN_PATH)
	passage_dict = id_content_reader(PASSAGE_PATH)
	return positive_dict, negative_dict, query_dict, passage_dict