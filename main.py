from network import DSSM
from train import train
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Current device is: %s" % torch.cuda.get_device_name(DEVICE))

net = DSSM().to(DEVICE)

def main(NUM_EPOCHS, EPOCH_SIZE):
    for ep_idx in range(NUM_EPOCHS):
        train_loss = train(EPOCH_SIZE)
        print(str(ep_idx) + ": " + str(train_loss))