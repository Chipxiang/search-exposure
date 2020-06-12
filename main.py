from network import DSSM
from train import train
import torch

NUM_EPOCHS = 5
EPOCH_SIZE = 10

def main(NUM_EPOCHS, EPOCH_SIZE):
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
	    print("Current device is: %s" % torch.cuda.get_device_name(DEVICE))
	net = DSSM().to(DEVICE)
    for ep_idx in range(NUM_EPOCHS):
        train_loss = train(net, EPOCH_SIZE)
        print(str(ep_idx) + ": " + str(train_loss))