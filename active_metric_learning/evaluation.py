import sys

sys.path.insert(0, '/home/jianx/search-exposure/')
import torch
from annoy import AnnoyIndex
import forward_ranker.load_data as load_data
import forward_ranker.train as train
from forward_ranker.utils import print_message
from forward_ranker.test import get_ndcg_precision_rr
obj_reader = load_data.obj_reader
obj_writer = load_data.obj_writer
import pickle
import csv
import numpy as np
import pandas as pd
import random
import math
import  matplotlib.pyplot as plt
from opts import get_opts
from testing import testing

OUTPUT_PATH = "/datadrive/ruohan/final_deliverable/all_results.csv"

def transform_ground_truth(true_dict, p):
    transform_true_dict = {}
    for pid, qid_rank in true_dict.items():
        transform_true_dict[pid] = {}
        for qid, rank in qid_rank.items():
            transform_true_dict[pid][qid] = p ** rank
    return transform_true_dict

def get_reverse_nrbp_rr(true_dict, test_dict, rank, p):
    sorted_result = list(test_dict.items())
    rank = min(rank, len(sorted_result))
    cumulative_gain = 0
    rr = float("NaN")
    for i in range(len(sorted_result)):
        pid = sorted_result[i][0]
        if pid in true_dict:
            rr = 1 / (i + 1)
            break
    for i in range(rank):
        pid = sorted_result[i][0]
        relevance = 0
        if pid in true_dict:
            relevance = true_dict[pid]
        discounted_gain = relevance * (p ** i)
        cumulative_gain += discounted_gain
    sorted_ideal = sorted(true_dict.items(), key=lambda x: x[1], reverse=True)
    ideal_gain = 0
    for i in range(rank):
        relevance = 0
        if i < len(sorted_ideal):
            relevance = sorted_ideal[i][1]
        discounted_gain = relevance * (p ** i)
        ideal_gain += discounted_gain
    nrbp = 0
    if ideal_gain != 0:
         nrbp = cumulative_gain / ideal_gain
    return nrbp, rr

def calculate_metrics(rating_dict, result_dict, rank, p):
    pids = list(result_dict.keys())
    result_nrbp = []
    result_rr = []
    for pid in pids:
        if pid in rating_dict:
            nrbp, rr = get_reverse_nrbp_rr(rating_dict[pid], result_dict[pid], rank, p)
            result_nrbp.append(nrbp)
            result_rr.append(rr)
    avg_nrbp = np.nanmean(result_nrbp)
    avg_rr = np.nanmean(result_rr)
    print("NRBP@{}: {:.4f}".format(rank,avg_nrbp), "RR: {:.4f}".format(avg_rr))
    return avg_nrbp

def grid_nrbp(p_forwards = [0.5,0.9,1], p_reverses = [0.5,0.9,1], ranks = [100]):
    x = []
    y = []
    data = []
    baseline_data = []
    opts = get_opts()
    ## Need to specify following arguments
    # reverse_ranker_path
    # test_data_path
    # device
    # active_learning_stage
    true_dict, baseline_dict, result_dict, args_dict = testing(opts)
    active_learning = args_dict["active_learning"]
    network_type = args_dict["network_type"]
    num_query = args_dict["num_query"]
    num_passage = args_dict["num_passage"]
    for r in ranks:
        for p_forward in p_forwards:
            rating_dict = transform_ground_truth(true_dict, p_forward)
            for p_reverse in p_reverses:
                x.append(p_forward)
                y.append(p_reverse)
                baseline_nrbp = calculate_metrics(rating_dict, baseline_dict, r, p_reverse)
                model_nrbp = calculate_metrics(rating_dict, result_dict, r, p_reverse)
                # data.append((model_nrbp-baseline_nrbp)/baseline_nrbp)
                data.append(model_nrbp)
                baseline_data.append(baseline_nrbp)
                print_message("Processed p_forward={}, p_reverse={}".format(p_forward, p_reverse))

    # Write results to csv
    output_results = [active_learning, network_type, num_query, num_passage] + data + baseline_data
    with open(OUTPUT_PATH, mode='a+') as output:
        output_writer = csv.writer(output)
        output_writer.writerow(output_results)

if __name__ == '__main__':
    grid_nrbp()
