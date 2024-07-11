import torch
from torch import nn, einsum
from einops import rearrange, repeat

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

List = nn.ModuleList

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual

class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask = None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward

def FeedForward(dim, ff_mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )

# classes

class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        edge_dim = None,
        heads = 8,
        gated_residual = True,
        with_feedforwards = False,
        norm_edges = False,
        rel_pos_emb = False,
        accept_adjacency_matrix = False
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        self.adj_emb = nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for _ in range(depth):
            self.layers.append(List([
                List([
                    PreNorm(dim, Attention(dim, pos_emb = pos_emb, edge_dim = edge_dim, dim_head = dim_head, heads = heads)),
                    GatedResidual(dim)
                ]),
                List([
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))

    def forward(
        self,
        nodes,
        edges = None,
        adj_mat = None,
        mask = None
    ):
        batch, seq, _ = nodes.shape

        if exists(edges):
            edges = self.norm_edges(edges)

        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(adj_mat.long())

        all_edges = default(edges, 0) + default(adj_mat, 0)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, all_edges, mask = mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges

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
    
    data = torch.load('tgeomdata/0.torch')
    
    from torch_geometric.utils import to_scipy_sparse_matrix
    import numpy as np
    
    # Create a sample graph
    #node_features = data.x#torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float)
    #edge_index = data.edge_index# torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Create a Data object
    #data = Data(x=node_features, edge_index=edge_index)
    model = GraphTransformer(
        dim = 1024,
        depth = 6,
        #edge_dim = 512,
        with_feedforwards = True,
        gated_residual = True,
        rel_pos_emb = True,
        accept_adjacency_matrix = True  # set this to True
    )
    
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
