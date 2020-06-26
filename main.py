from network import DSSM
from train import train
from load_data import load
from gpu_allocator import select_device
from gpu_allocator import cleanup_gpu_list
import torch
import csv
from test import test
import sys
import os

NUM_EPOCHS = int(sys.argv[1])
EPOCH_SIZE = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])
EMBED_SIZE = int(sys.argv[5])
GPU_ROOT = "/home/jianx/data/gpu_usage.list"

CURRENT_GPU_ID, CURRENT_DEVICE = select_device(GPU_ROOT)
print(CURRENT_DEVICE)
print("Num of epochs:", NUM_EPOCHS)
print("Epoch size:", EPOCH_SIZE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Embedding size:", EMBED_SIZE)
RANK = 10
TEST_BATCH = 43
MODEL_PATH = "/home/jianx/data/results/"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path, rank, test_batch, embed_size):
    net = DSSM(embed_size=embed_size).to(CURRENT_DEVICE)
    print("Loading data")
    pos_neg_dict, query_dict, passage_dict, top_dict, rating_dict, query_test_dict = load()
    print("Data successfully loaded.")
    print("Positive Negative Pair dict size: " + str(len(pos_neg_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))

    arg_str = str(num_epochs) + "_" + str(epoch_size) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(embed_size)
    unique_path = model_path + arg_str + ".model"
    output_path = model_path + arg_str + ".csv"
    for ep_idx in range(num_epochs):
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        train_loss = train(net, epoch_size, batch_size, optimizer, CURRENT_DEVICE, pos_neg_dict,
                           query_dict, passage_dict)
        avg_ndcg, avg_prec, avg_rr = test(net, CURRENT_DEVICE, test_batch, top_dict, query_test_dict, passage_dict,
                                          rating_dict,
                                          rank)
        print("Epoch:{}, loss:{}, NDCG:{}, P:{}, RR:{}".format(ep_idx, train_loss, avg_ndcg, avg_prec, avg_rr))
        with open(output_path, mode='a+') as output:
            output_writer = csv.writer(output)
            output_writer.writerow([ep_idx, train_loss, avg_ndcg, avg_prec, avg_rr])
    torch.save(net, unique_path)
    cleanup_gpu_list(CURRENT_GPU_ID, GPU_ROOT)


if __name__ == '__main__':
    main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, RANK, TEST_BATCH, EMBED_SIZE)
