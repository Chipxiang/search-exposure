from network import DSSM
from train import train
from load_data import load

NUM_EPOCHS = 5
EPOCH_SIZE = 5
BATCH_SIZE = 10
LEARNING_RATE = 0.001


def main(num_epochs, epoch_size, batch_size, learning_rate):
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


if __name__ == '__main__':
    main(NUM_EPOCHS, EPOCH_SIZE, BATCH_SIZE, LEARNING_RATE)
