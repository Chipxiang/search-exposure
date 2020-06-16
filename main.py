from network import DSSM
from train import train
from load_data import load
import torch
import csv
from datetime import datetime
from test import test
import sys

NUM_EPOCHS = int(sys.argv[1])
EPOCH_SIZE = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])
print("Num of epochs:", NUM_EPOCHS)
print("Epoch size:", EPOCH_SIZE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
RANK = 10
TEST_BATCH = 35
MODEL_PATH = "/home/jianx/data/"


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path, rank, test_batch):
    dssm = DSSM()
    net = dssm.to(dssm.device)
    print("Loading data")
    pos_neg_dict, query_dict, passage_dict, top_dict, rating_dict, query_test_dict = load()
    print("Data successfully loaded.")
    print("Positive Negative Pair dict size: " + str(len(pos_neg_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))
    # date_time_obj = datetime.now()
    # timestamp_str = date_time_obj.strftime("%b-%d-%Y_%H:%M:%S")
    arg_str = str(NUM_EPOCHS) + "_" + str(EPOCH_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE)
    unique_path = model_path + arg_str + ".model"
    output_path = model_path + arg_str + ".csv"
    with open(output_path, mode='a') as output:
        output_writer = csv.writer(output)
        for ep_idx in range(num_epochs):
            train_loss = train(net, epoch_size, batch_size, learning_rate, dssm.device, pos_neg_dict,
                               query_dict, passage_dict)
            avg_ndcg, avg_prec, avg_rr = test(net, test_batch, top_dict, query_test_dict, passage_dict, rating_dict,
                                              rank)
            print("Epoch:{}, loss:{}, NDCG:{}, P:{}, RR:{}".format(ep_idx, train_loss, avg_ndcg, avg_prec, avg_rr))
            output_writer.writerow([ep_idx, train_loss, avg_ndcg, avg_prec, avg_rr])
    torch.save(net, unique_path)


if __name__ == '__main__':
    main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, RANK, TEST_BATCH)
