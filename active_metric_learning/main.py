import torch
from torch import optim
import csv
import sys
import os

from util import print_message
from network import ResidualNet, AppendNet
from train import train
from opts import get_opts
from load_data import load



def main():
    opts = get_opts()
    # Paths and device
    current_device = opts.device
    train_data_path = opts.data_dir
    pretrained_path = opts.pretrain_model_path
    model_path = opts.out_dir
    # training settings
    num_epochs = opts.num_epochs
    learning_rate = opts.learning_rate
    num_query = opts.num_query
    num_passage = opts.num_passage
    active_learning = opts.active_learning_stage
    # network settings
    network_type = opts.network_type
    embed_size = opts.embed_size
    num_hidden_nodes = opts.num_hidden_nodes
    num_hidden_layers = opts.num_hidden_layers
    dropout_rate = opts.dropout_rate

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if pretrained:
        checkpoint = torch.load(pretrained_path)
        network_type = checkpoint['network_type']
        embed_size = checkpoint['embed_size']
        num_hidden_nodes = checkpoint['num_hidden_nodes']
        num_hidden_layers = checkpoint['num_hidden_nodes']
        dropout_rate = checkpoint['dropout_rate']
        if network_type == "append":
            net = AppendNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                            num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
        if network_type == "residual":
            net = ResidualNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                            num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
        net.load_state_dict(checkpoint['model'])
        net.to(current_device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        if network_type == "append":
            net = AppendNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                    num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate).to(current_device)
        if network_type == "residual":
            net = ResidualNet(embed_size=embed_size, num_hidden_nodes=num_hidden_nodes, 
                    num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate).to(current_device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print("Loading data")
    train_pos_dict, train_neg_dict, query_dict, passage_dict = load(train_data_path)
    print("Data successfully loaded.")
    print("Negative Pair dict size: " + str(len(train_neg_dict)))
    print("Positive Pair dict size: " + str(len(train_pos_dict)))
    print("Num of queries: " + str(len(query_dict)))
    print("Num of passages: " + str(len(passage_dict)))
    print("Finish loading.")

    arg_str = active_learning + network_type + str(num_query) + "query" + str(num_passage) + "passage"
    unique_path = model_path + arg_str + ".model"
    output_path = model_path + arg_str + ".csv"
    
    for ep_idx in range(num_epochs):
        train_loss = train(net, optimizer, opts, train_pos_dict, 
                           train_neg_dict, query_dict, passage_dict)
        print_message([ep_idx,train_loss])
        with open(output_path, mode='a+') as output:
            output_writer = csv.writer(output)
            output_writer.writerow([ep_idx, train_loss])
        torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_epoch": ep_idx,
                "train_loss": train_loss,
                "network_type": network_type,
                "embed_size": embed_size,
                "num_hidden_nodes": num_hidden_nodes,
                "n_hidden_layer": num_hidden_layers,
                "dropout_rate": dropout_rate
                    }, unique_path)

if __name__ == '__main__':
    main()