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
import math

OUTPUT_PATH = "/datadrive/ruohan/final_results_list/"

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
    result_nrbp = {}
    for pid in pids:
        if pid in rating_dict:
            nrbp, rr = get_reverse_nrbp_rr(rating_dict[pid], result_dict[pid], rank, p)
            if math.isnan(nrbp):
                continue
            result_nrbp[pid] = nrbp
    print(len(result_nrbp))
    return result_nrbp

def grid_nrbp(p_forwards = [0.5], p_reverses = [0.9], ranks = [100]):
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
    results = {}
    for r in ranks:
        for p_forward in p_forwards:
            rating_dict = transform_ground_truth(true_dict, p_forward)
            for p_reverse in p_reverses:
                baseline_nrbp = calculate_metrics(rating_dict, baseline_dict, r, p_reverse)
                model_nrbp = calculate_metrics(rating_dict, result_dict, r, p_reverse)
                results["baseline_nrbp"] = baseline_nrbp
                results["model_nrbp"] = model_nrbp
    obj_writer(results, OUTPUT_PATH + network_type + ".dict")

if __name__ == '__main__':
    grid_nrbp()