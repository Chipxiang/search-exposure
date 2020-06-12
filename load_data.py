import csv

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