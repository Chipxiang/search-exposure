import torch
import torch.nn as nn

# NUM_HIDDEN_NODES = 1536
# NUM_HIDDEN_LAYERS = 1
# DROPOUT_RATE = 0.1
# embed_size = 768
    
# Define the residual network
class ResidualNet(torch.nn.Module):

    def __init__(self, embed_size, num_hidden_nodes, num_hidden_layers, dropout_rate):
        super(ResidualNet, self).__init__()
        
        self.input = nn.Linear(embed_size, num_hidden_nodes)
        self.relu = nn.ReLU()
        self.normlayer = nn.LayerNorm(num_hidden_nodes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(num_hidden_nodes, embed_size)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x):
        identity = x
        out = x
        for i in range(self.num_hidden_layers):
            out = self.input(out)
            out = self.relu(out)
            out = self.normlayer(out)
            out = self.dropout(out)
            out = self.output(out)
            out += identity
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Define the appending network
# NUM_HIDDEN_NODES = 64
# NUM_HIDDEN_LAYERS = 3
# DROPOUT_RATE = 0.1
# embed_size = 32
FEAT_COUNT = 768
class AppendNet(torch.nn.Module):

    def __init__(self, embed_size, num_hidden_nodes, num_hidden_layers, dropout_rate):
        super(AppendNet, self).__init__()

        layers = []
        last_dim = FEAT_COUNT
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, num_hidden_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(num_hidden_nodes))
            layers.append(nn.Dropout(p=dropout_rate))
            last_dim = num_hidden_nodes
        layers.append(nn.Linear(last_dim, embed_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = torch.cat((x, self.model(x)), 1)
        return output

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
