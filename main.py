from network import DSSM
from train import train
from load_data import load
import torch
from datetime import datetime

NUM_EPOCHS = 20
EPOCH_SIZE = 750
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
MODEL_PATH = "/home/jianx/data/"


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path):
    dssm = DSSM()
    net = dssm.to(dssm.device)
    print("Loading data")
    pos_neg_dict, query_dict, passage_dict, top_dict, rating_dict, query_test_dict = load()
    print("Data successfully loaded.")
    print("Positive Negative Pair dict size: " + str(len(pos_neg_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))
    for ep_idx in range(num_epochs):
        train_loss = train(net, epoch_size, batch_size, learning_rate, dssm.device, pos_neg_dict,
                           query_dict, passage_dict)
        print(str(ep_idx) + ": " + str(train_loss))
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%%b-%d-%Y_%H:%M:%S")
    unique_path = model_path + timestamp_str + ".model"
    torch.save(net, unique_path)


if __name__ == '__main__':
    main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE, MODEL_PATH)
