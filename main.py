from network import DSSM
from train import train
from load_data import load
import torch

NUM_EPOCHS = 50
EPOCH_SIZE = 1000
BATCH_SIZE = 200
LEARNING_RATE = 0.001
MODEL_PATH = "/home/jianx/data/trained_model.model"


def main(num_epochs, epoch_size, batch_size, learning_rate, model_path):
    dssm = DSSM()
    net = dssm.to(dssm.device)
    print("Loading data")
    positive_dict, negative_dict, query_dict, passage_dict = load()
    print("Data successfully loaded.")
    print("Positive dict size: " + str(len(positive_dict)))
    print("Negative dict size: " + str(len(negative_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))
    for ep_idx in range(num_epochs):
        train_loss = train(net, epoch_size, batch_size, learning_rate, dssm.device, positive_dict, negative_dict,
                           query_dict, passage_dict)
        print(str(ep_idx) + ": " + str(train_loss))
    torch.save(net, model_path)


if __name__ == '__main__':
    main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE, MODEL_PATH)
