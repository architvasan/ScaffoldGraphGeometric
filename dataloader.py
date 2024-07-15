import torch
from torch import nn, einsum
from einops import rearrange, repeat
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from tqdm import tqdm
# Import plotting tools
import matplotlib.pyplot as plt
# Import rdkit
from rdkit.Chem import Draw
from rdkit import Chem
# import libraries
import torch
import torch_geometric.data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from itertools import chain, repeat, islice
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import random
import os
from torch.utils.data import Dataset
from argparse import ArgumentParser, SUPPRESS
import selfies
from graphtransf_encoder import *
if False:
    model = GraphTransformer(
        dim = 256,
        depth = 6,
        edge_dim = 512,             # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
        with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
        gated_residual = True,      # to use the gated residual to prevent over-smoothing
        rel_pos_emb = True          # set to True if the nodes are ordered, default to False
    )
    model.to("cuda")

#nodes = torch.randn(1, 128, 256)
#edges = torch.randn(1, 128, 128, 512)
#mask = torch.ones(1, 128).bool()

#nodes, edges = model(nodes, edges, mask = mask)

#print(nodes.shape) # (1, 128, 256) - project to R^3 for coordinates

class Custom_Dataset(Dataset):
    def __init__(self, tokenizer, inpdata, max_length, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.data = inpdata

        # Tokenize the entire dataset during initialization
        self.inputs = self.tokenizer(
            self.data,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Transfer to device
        self.inputs["input_ids"] = self.inputs["input_ids"].to(self.device)
        self.inputs["attention_mask"] = self.inputs["attention_mask"].to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.inputs["input_ids"][idx]  # Assuming labels are same as input_ids
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }







# Create a sample graph
#node_features = data.x#torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float)
#edge_index = data.edge_index# torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

# Create a Data object
#data = Data(x=node_features, edge_index=edge_index)
model = GraphTransformer(
    dim = 1024,
    depth = 6,
    edge_dim = 909,
    with_feedforwards = True,
    gated_residual = True,
    rel_pos_emb = True,
    accept_adjacency_matrix = True  # set this to True
)
#model.to("cuda")
model.eval()

data = torch.load('tgeomdata/0.torch')
#nodes = torch.randn(2, 128, 256)
#adj_mat = torch.randint(0, 2, (2, 128, 128))
mask = torch.ones(1, data.x.shape[0]).bool()
mask.to("cuda")

# Convert edge_index to a SciPy sparse adjacency matrix
adj_matrix_sparse = to_scipy_sparse_matrix(data.edge_index)
# Convert the sparse matrix to a dense numpy array (optional)
adj_matrix_dense = torch.from_numpy(adj_matrix_sparse.todense())
# Unsqueeze the tensor to add a new dimension at index 0, resulting in (1, 573, 573)
adj_matrix_dense = torch.unsqueeze(adj_matrix_dense, 0)
adj_matrix_dense.to("cuda")
print(adj_matrix_dense)
print(adj_matrix_dense.shape)

node_tensor = torch.unsqueeze(data.x, 0)
node_tensor.to("cuda")
# Access the node features and edge indices
print("Node Features:")
print(node_tensor.shape)
print("Edge Indices:")
print(data.edge_index.shape)

nodes_new, edges_new = model(node_tensor, adj_mat = adj_matrix_dense, mask = mask)


print("nodes and edges from model")
print(nodes_new.shape)
#print(edges_new.shape)
print(nodes_new)
print(edges_new)
#nodes.shape # (1, 128, 256) - project to R^3 for coordinates
