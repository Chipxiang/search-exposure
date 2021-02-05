ground_truth_path = "/datadrive/ruohan/bm25/ground_truth/bm25_top100.dict"
import pickle
import argparse

def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")

def get_opts():
    parser = argparse.ArgumentParser(description='Generate bm25 ranking results')  

    parser.add_argument('--baseline_path', type=str)
    parser.add_argument('--output_baseline_path', type=str)
    opts = parser.parse_args()
    return opts

def generate_test_data():
    opts = get_opts()
    baseline_path = opts.baseline_path
    output_baseline_path = opts.output_baseline_path
    print(baseline_path)
    all_results = obj_reader(ground_truth_path)
    # all_results = {}
    # with open(ground_truth_path, "r") as f:
    #     for line in f:
    #         line_split = line.split(",")
    #         qid = int(line_split[0])
    #         pid = int(line_split[1])
    #         rank = int(line_split[2])
    #         if qid not in all_results.keys():
    #             all_results[qid] = {}
    #         all_results[qid][pid] = rank
    with open(baseline_path, "r") as f:
        for line in f:
            line_split = line.split("\t")
            pid = int(line_split[0])
            qid = int(line_split[1])
            rank = all_results[qid].get(pid, 0)
            with open(output_baseline_path, "a") as f:
                f.write("{},{},{}\n".format(pid, qid, rank))

if __name__ == '__main__':
    generate_test_data()
