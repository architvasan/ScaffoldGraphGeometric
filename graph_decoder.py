import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import random

class GraphTransformerDecoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers, num_heads):
        super(GraphTransformerDecoder, self).__init__()
        self.node_embedding = nn.Embedding(num_node_features, hidden_dim)
        self.edge_embedding = nn.Embedding(num_edge_features, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim, num_node_features + num_edge_features)

    def forward(self, src, tgt, memory_mask=None, tgt_mask=None):
        src = self.node_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.node_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        
        output = self.transformer_decoder(tgt, src, memory_mask=memory_mask, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

def graph_to_sequence(data):
    nodes = data.x.view(-1)
    edges = data.edge_index.t().reshape(-1)
    sequence = torch.cat([nodes, edges], dim=0)
    return sequence

data_list = [
    Data(x=torch.tensor([[1], [2], [3], [4]], dtype=torch.long),
         edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
         y=torch.tensor([0], dtype=torch.long))
]

sequences = [graph_to_sequence(data) for data in data_list]
print(sequences)
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        src = sequence[:-1]
        tgt = sequence[1:]
        return src, tgt

dataset = SequenceDataset(sequences)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphTransformerDecoder(num_node_features=10, num_edge_features=10, hidden_dim=64, num_layers=2, num_heads=4)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        output = model(src, tgt)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

def generate_sequence(model, start_token, max_length=100):
    model.eval()
    sequence = [start_token]
    for _ in range(max_length):
        src = torch.tensor([sequence], device=device)
        tgt = torch.tensor([sequence], device=device)

        output = model(src, tgt)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        sequence.append(next_token)
        if next_token == end_token:  # Assuming end_token is defined
            break
    return sequence

start_token = 0  # Assuming 0 is the start token
end_token = 1    # Assuming 1 is the end token
generated_sequence = generate_sequence(model, start_token)
print(generated_sequence)

