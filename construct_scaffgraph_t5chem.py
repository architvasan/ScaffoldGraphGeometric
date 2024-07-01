import scaffoldgraph as sg
# Import networkx
import networkx as nx
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

class Custom_Dataset(Dataset):
        def __init__(self, tokenizer, inpdata, max_length, batch, device="cuda"):
            self.tokenizer = tokenizer
            self.device = device
            self.max_length = max_length
            self.data = inpdata
            self.batch = batch

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data = str(self.data[idx])
            labels = str(self.data[idx])

            # tokenize data
            inputs = self.tokenizer(data, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = self.tokenizer(labels, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

            return {
                "input_ids": inputs["input_ids"].flatten().to(self.device),
                "attention_mask": inputs["attention_mask"].flatten().to(self.device),
                "labels": labels["input_ids"].flatten().to(self.device),
            }

def dataloader(customdata, batch):
    return torch.utils.data.DataLoader(customdata, batch_size=batch, shuffle=False)

def encode_nodes(dataloader, LLModel):
    node_data_encoded = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels) 
            encoder = outputs["encoder_last_hidden_state"][-1]
            encoder = encoder.mean(dim=1).detach().cpu().numpy()
        node_data_encoded.append([encoder])
    return torch.tensor(np.array(node_data_encoded))

def ScaffoldGraphStructure(inp_smi_file, tokenizer, LLModel, batch):
    network = sg.ScaffoldNetwork.from_smiles_file(inp_smi_file, progress=True)
    nodes = list(network.nodes())
    nodes_raw = np.array(nodes)
    #print(nodes_raw)
    nodes = [s.replace("MolNode-", "") for s in nodes]
    edges_raw = list(network.edges())
    edges_indices = []
    for edge in tqdm(edges_raw):
        it_0 = np.where(nodes_raw==edge[0])
        it_1 = np.where(nodes_raw==edge[1])
        edges_indices.append([list(list(it_0)[0])[0], list(list(it_1)[0])[0]])

    edge_tensor = torch.tensor(np.array(edges_indices), dtype=torch.long)

    ScaffGraphData = Custom_Dataset(tokenizer, nodes, max_length, batch, device="cuda")
    scaffdataload = dataloader(ScaffGraphData, batch)#ScaffGraphOps.dataload()
    encoded_nodes = encode_nodes(scaffdataload, LLModel)
    print(encoded_nodes)
    graph_structure = torch_geometric.data.Data(x=encoded_nodes, edge_index=edge_tensor.t().contiguous())
    return graph_structure


################################################################################
################################################################################
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-s", "--smifile", type=str, required=True, help="Input data for training"
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
LLModel = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
LLModel.to("cuda")
LLModel.eval()

#inp_smi_file = '3CLPro_test.smi'
max_length = 512
batch = 1

ScaffGraphStruct = ScaffoldGraphStructure(args.smifile, tokenizer, LLModel, batch)
print(ScaffGraphStruct)

