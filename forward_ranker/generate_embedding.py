import sys
import torch

import network
from load_data import obj_writer
from load_data import obj_reader
from train import generate_sparse

EMBED_SIZE = 256
DEVICE = torch.device("cuda")


def generate_collection_embedding(net, passage_dict, device=torch.device("cuda")):
    embedding_dict = {}
    counter = 0

    for key, value in passage_dict.items():
        if len(value) != 0:
            embedding_dict[key] = net(generate_sparse(value).to(device)).detach().tolist()
        if counter % 10000 == 0:
            print("Generating embeddings: " + str(counter) + "/" + str(len(passage_dict)))
        counter += 1
    return embedding_dict


if __name__ == '__main__':
    model_path = sys.argv[1]
    target_dict = sys.argv[2]
    save_path = sys.argv[3]
    model = network.DSSM(embed_size=EMBED_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    print("Reading target dictionary.")
    passages = obj_reader(target_dict)
    embedding = generate_collection_embedding(model, passages)
    print("Saving embeddings dictionary")
    obj_writer(embedding, save_path)
