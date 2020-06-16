import torch
import torch.nn as nn


NUM_HIDDEN_NODES = 64
NUM_HIDDEN_LAYERS = 3
DROPOUT_RATE = 0.1
FEAT_COUNT = 50000


# Define the network
class DSSM(torch.nn.Module):

    def __init__(self):
        super(DSSM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Current device is: %s" % torch.cuda.get_device_name(self.device))
        self.scale = torch.tensor([1], dtype=torch.float).to(self.device)

        layers = []
        last_dim = FEAT_COUNT
        for i in range(NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(last_dim, NUM_HIDDEN_NODES))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(NUM_HIDDEN_NODES))
            layers.append(nn.Dropout(p=DROPOUT_RATE))
            last_dim = NUM_HIDDEN_NODES
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) * self.scale

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
