import torch
import torch.nn as nn
import numpy as np
import faiss
import os
from util import obj_reader, obj_writer
from network import ResidualNet, AppendNet
from opts import get_opts


PASSAGE_NP_PATH = "/home/jianx/results/passage_0__emb_p__data_obj_0.pb"
PASSAGE_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_passage_map.dict"
QUERY_TRAIN_NP_PATH = "/home/jianx/results/query_0__emb_p__data_obj_0.pb"
QUERY_TEST_NP_PATH = "/home/jianx/results/test_query_0__emb_p__data_obj_0.pb"
QUERY_MAP_PATH = "/datadrive/jianx/data/annoy/100_ance_query_train_map.dict"
TRUE_PATH = "/datadrive/jianx/data/results/all_search_rankings_100_100_flat.csv"
QUERY_DEV_NP_PATH = "/home/jianx/results/dev_query_0__emb_p__data_obj_0.pb"

TRAIN_RANK_PATH = "/datadrive/jianx/data/train_data/ance_testing_rank100_nqueries50000_20000_Sep_03_22:56:31.csv"
REVERSE_RANKER_PATH = "/datadrive/ruohan/fix_residual_overfit/reverse_alpha0.5_layer1_residual1000_100_1000_0.0001_768.model"

# Load data
def load_true_dict(k = 100, path = TRUE_PATH):
    true_dict = {}
    with open(path, "r") as file:
        for line in file:
            qid = int(line.split(",")[0])
            pid = int(line.split(",")[1])
            rank = int(line.split(",")[2])
            if rank > k:
                continue
            if pid not in true_dict.keys():
                true_dict[pid] = {}
            true_dict[pid][qid] = rank
    return true_dict

def load_forward_dict(k = 100, path = TRUE_PATH):
    true_dict = {}
    with open(path, "r") as file:
        for line in file:
            qid = int(line.split(",")[0])
            pid = int(line.split(",")[1])
            rank = int(line.split(",")[2])
            if rank > k:
                continue
            if qid not in true_dict.keys():
                true_dict[qid] = {}
            true_dict[qid][pid] = rank
    return true_dict

def load_train(path):
    with open(path) as file:
        my_dict = {}
        count = 0
        for line in file:
            count += 1
            tokens = line.split(",")
            pid = int(tokens[0])
            qid = int(tokens[1])
            rank = int(tokens[2].rstrip())
            if pid not in my_dict:
                my_dict[pid] = {}
            my_dict[pid][qid] = rank
    return my_dict

# Load model
def load_model(path):
    checkpoint = torch.load(path)
    network_type = checkpoint['network_type']
    embed_size = checkpoint['embed_size']
    num_hidden_nodes = checkpoint['num_hidden_nodes']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']
    num_query = checkpoint['num_query']
    num_passage = checkpoint['num_passage']
    if network_type == "append":
        net = AppendNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                        num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    if network_type == "residual":
        net = ResidualNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                        num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    net.load_state_dict(checkpoint['model'])
    net.to(current_device)
    net.eval()
    return net, network_type    

def transform_np_transformation(query_np, reverse_ranker, device, b=10000):
    n = int(query_np.shape[0]/b) + 1

    corpus_output = []
    for i in range(n):
        start = i * b
        end = (i + 1) * b
        if i == n-1:
            end = query_np.shape[0]
        q_embed = query_np[start:end,:]
        q_embed = torch.from_numpy(q_embed).to(device)
        corpus_output.append(reverse_ranker(q_embed).detach().cpu().numpy())
    corpus_np = np.concatenate(corpus_output[:-1])
    corpus_np = np.concatenate((corpus_np, corpus_output[-1]))
    print(corpus_np.shape)
    return corpus_np

# Generate testing results for reverse ranker

def generate_pred_rank(query_index, true_dict, baseline_dict, passage_embed, 
                       qid_mapping, pid_reverse_mapping, n=100, k=100):
    pid_list = list(baseline_dict.keys())
    p_embed_list = []
    all_results = {}
    print("Begin append.")
    for i, pid in enumerate(pid_list):
        if i >= n:
            break
        pid_r = pid_reverse_mapping[pid]
        p_embed = np.array(passage_embed[pid_r])
        p_embed_list.append(p_embed)
    p_embed_all = np.stack(p_embed_list)
    print("Finish append.")
    print("Begin search.")
    _, near_qids = query_index.search(p_embed_all, k)
    print("Finish search.")
    for i, pid in enumerate(pid_list):
        temp_results = {}
        for qid in near_qids[i]:
            qid = qid_mapping[qid]
            try:
                rank = true_dict[pid][qid]
            except:
                rank = 0
            temp_results[qid] = rank
        all_results[pid] = temp_results
    return all_results

def evaluate_reverse_ranker(pred_rank, true_rank, k = 100):
    top_true = []
    top_pred = []
    for pid, qids in pred_rank.items():
        n_top_true = len(true_rank.get(pid, {}))
        temp_pred = np.fromiter(qids.values(), dtype=int)
        n_top_pred = sum((temp_pred != 0) & (temp_pred <= k))
        top_true.append(n_top_true)
        top_pred.append(n_top_pred)
    return top_true, top_pred

def compare_with_baseline(query_index, true_dict_100, forward_baseline_rank, passage_embed, 
                          qid_mapping, pid_reverse_mapping,n):
    pred_rank = generate_pred_rank(query_index, true_dict_100, forward_baseline_rank, 
                                   passage_embed, qid_mapping, pid_reverse_mapping, n=n)
    top_true, top_pred = evaluate_reverse_ranker(pred_rank, true_dict_100, k = 100)
    print("New model: {}".format(np.mean(top_pred)/np.mean(top_true)))
    top_true_baseline, top_pred_baseline = evaluate_reverse_ranker(forward_baseline_rank, true_dict_100, k = 100)
    print("Baseline model: {}".format(np.mean(top_pred_baseline)/np.mean(top_true_baseline)))
    return top_true, top_pred, top_true_baseline, top_pred_baseline, pred_rank

def delete_zeros(myDict):
    out_dict = {key:val for key, val in myDict.items() if val != 0}
    return set(list(out_dict.keys()))

def compare_specific_passage(pred_rank_test1, forward_baseline_rank_test1, n):
    count = 0
    count_loss = 0
    all_count = 0
    all_count_loss = 0
    for i in range(n):
        pid = list(pred_rank_test1.keys())[i]
        pred = delete_zeros(pred_rank_test1[pid])
        baseline = delete_zeros(forward_baseline_rank_test1[pid])
        diff = pred - baseline
        if diff != set():
            count += 1
            all_count += len(diff)
        diff_loss = baseline - pred
        if diff_loss != set():
            count_loss += 1
            all_count_loss += len(diff_loss)
    print("Percentage of passages with newly found exposing queries: {}".format(count/n))
    print("Percentage of passages that lose originaly found exposing queries: {}".format(count_loss/n))
    print("Number of newly found exposing queries:{} Number of lost exposing queries:{} Net gain:{}".format(all_count, all_count_loss, all_count - all_count_loss))

# Compute NRBP


# Main function
def testing(opts):
    reverse_ranker_path = opts.reverse_ranker_path
    test_data_path = opts.test_data_path
    test_output_path = opts.test_output_path
    current_device = opts.device
    active_learning = opts.active_learning_stage

    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    # Load data
    print("Load passages.")
    passage_np = obj_reader(PASSAGE_NP_PATH)
    pid_mapping = obj_reader(PASSAGE_MAP_PATH)
    print("Load queries.")
    query_np = obj_reader(QUERY_TRAIN_NP_PATH)
    qid_mapping = obj_reader(QUERY_MAP_PATH)
    print("Load pre-processed results.")
    true_dict_100 = load_true_dict(k=100)
    pid_reverse_mapping = {v: k for k, v in pid_mapping.items()}

    # Load model
    checkpoint = torch.load(reverse_ranker_path)
    network_type = checkpoint['network_type']
    embed_size = checkpoint['embed_size']
    num_hidden_nodes = checkpoint['num_hidden_nodes']
    num_hidden_layers = checkpoint['num_hidden_layers']
    dropout_rate = checkpoint['dropout_rate']
    num_query = checkpoint['num_query']
    num_passage = checkpoint['num_passage']
    if network_type == "append":
        net = AppendNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                        num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    if network_type == "residual":
        net = ResidualNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                        num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    net.load_state_dict(checkpoint['model'])
    net.to(current_device)
    net.eval()

    # Data preparation
    query_new_np = transform_np_transformation(query_np, net, current_device)
    passage_new_np = transform_np_transformation(passage_np, net, current_device)
    dim = query_new_np.shape[1]
    query_index = faiss.IndexFlatIP(dim)
    query_index.add(query_new_np)

    # Generate testing data
    forward_baseline_rank_test = load_train(test_data_path)
    top_true_test, top_pred_test, top_true_baseline_test, top_pred_baseline_test, pred_rank_test = compare_with_baseline(query_index, 
        true_dict_100, forward_baseline_rank_test, passage_new_np, qid_mapping, pid_reverse_mapping, n=20000)
    compare_specific_passage(pred_rank_test, forward_baseline_rank_test, n=20000)

    # Write results to dict
    results_dict = {}
    results_dict['forward_baseline_rank_test'] = forward_baseline_rank_test
    results_dict['pred_rank_test'] = pred_rank_test
    output_path = active_learning + "_" + network_type + "_" + str(num_query) + "_"  + "query" + "_" + str(num_passage) + "_"  + "passage" + ".dict"
    obj_writer(results_dict, test_output_path + output_path)

    args_dict = {"active_learning": active_learning, "network_type":network_type,
                "num_query": num_query, "num_passage": num_passage}

    return true_dict_100, forward_baseline_rank_test, pred_rank_test, args_dict




