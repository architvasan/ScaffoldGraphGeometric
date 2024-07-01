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
from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
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

class Custom_Dataset(Dataset):
        def __init__(self, tokenizer, inpdata, max_length, batch, device="cuda"):
            self.tokenizer = tokenizer
            self.device = device
            self.max_length = max_length
            #print(network.nodes())
            self.data = inpdata#list(self.network.nodes())
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
            #print(outputs.shape)
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
        #print(list(it_0)[0])
        #print(list(it_1)[0])
        edges_indices.append([list(list(it_0)[0])[0], list(list(it_1)[0])[0]])

    edge_tensor = torch.tensor(np.array(edges_indices), dtype=torch.long)

    ScaffGraphData = Custom_Dataset(tokenizer, nodes, max_length, batch, device="cuda")
    scaffdataload = dataloader(ScaffGraphData, batch)#ScaffGraphOps.dataload()
    encoded_nodes = encode_nodes(scaffdataload, LLModel)
    print(encoded_nodes)
    graph_structure = torch_geometric.data.Data(x=encoded_nodes, edge_index=edge_tensor.t().contiguous())
    #edges = ""
    return graph_structure


################################################################################
################################################################################
tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
LLModel = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
LLModel.to("cuda")
LLModel.eval()

inp_smi_file = '3CLPro_test.smi'
max_length = 512
batch = 1

ScaffGraphStruct = ScaffoldGraphStructure(inp_smi_file, tokenizer, LLModel, batch)
print(ScaffGraphStruct)

#ScaffGraphData = ScaffoldGraph_Dataset(tokenizer, LLModel, inp_smi_file, max_length, batch, device="cuda")
#scaffdataload = dataloader(ScaffGraphData, batch)#ScaffGraphOps.dataload()
#encoded_nodes = encode_nodes(scaffdataload, LLModel)
##print(encoded_nodes)
















if False:
    #print(network.nodes())
    networknodes = network.nodes()
    node_tensors = []#np.array([])
    for nodes in list(networknodes):
        inputs = tokenizer(list(nodes), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        inputs.to('cuda')
        outputs = LLModel(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["input_ids"], return_tensors="pt")
        encoder = outputs["encoder_last_hidden_state"]
        encoder = encoder.mean(dim=1).detach().cpu().numpy()
        del(outputs)
        #print(encoder)
        node_tensors.extend(encoder)
        del(encoder)
        print(node_tensors)
    
    x = torch.tensor(np.array([[tokenize_function(i, 64)] for i in network.nodes()]))
    print(x)
    #x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    edge_index = torch.tensor(np.array([[0, 1], [1, 2]]), dtype=torch.long)

if False:
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index.t().contiguous())

## construct a scaffold tree from a pandas dataframe
#import pandas as pd
#df = pd.read_csv('activity_data.csv')
#network = sg.ScaffoldTree.from_dataframe(
#    df, smiles_column='Smiles', name_column='MolID',
#    data_columns=['pIC50', 'MolWt'], progress=True,
#)

if False:
    class CustomDataset(Dataset):
        def __init__(self, tokenizer, data, y_regression_values, max_input_length, max_target_length, device="cuda"):
            self.tokenizer = tokenizer
            self.data = data
            self.y_regression_values = y_regression_values
            self.device = device
            self.max_input_length = max_input_length
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data = str(self.data[idx])
            labels = str(self.data[idx])

            
            

            return {
                "input_ids": inputs["input_ids"].flatten().to(self.device),
                "attention_mask": inputs["attention_mask"].flatten().to(self.device),
                "labels": labels["input_ids"].flatten().to(self.device),
                "y_regression_values": torch.tensor(self.y_regression_values[idx]).to(self.device),
            }

        def test():
            target_train = smiles_df_train['labels'].values
            features_train = [tok_dat for tok_dat in smiles_df_train['text']]#np.stack([tok_dat for tok_dat in smiles_df_train['text']])
            smiles_df_val['text'] = smiles_df_val['text'].progress_apply(lambda x: tokenize_function(x, ntoken=ntoken))
            target_val = smiles_df_val['labels'].values
            features_val = [tok_dat for tok_dat in smiles_df_val['text']]#np.stack([tok_dat for tok_dat in smiles_df_val['text']])

            feature_tensor_train = torch.tensor(features_train)
            label_tensor_train = torch.tensor(smiles_df_train['labels'])
            feature_tensor_val = torch.tensor(features_val)
            label_tensor_val = torch.tensor(smiles_df_val['labels'])

            train_dataset = TensorDataset(feature_tensor_train, label_tensor_train)
            val_dataset = TensorDataset(feature_tensor_val, label_tensor_val)

            train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
            return train_dataloader, val_dataloader, val_dataset

        # construct a scaffold network from an SDF file
        #network = sg.ScaffoldNetwork.from_sdf('my_sdf_file.sdf')
        # construct a scaffold tree from a SMILES file
        network = sg.ScaffoldNetwork.from_smiles_file('3CLPro_test.smi', progress=True)
        #print(network)
        #object_methods = [method_name for method_name in dir(network)
        #                  if callable(getattr(network, method_name))]
        #print(object_methods)
        #print(network.edges())
        #print(network.nodes())
        
