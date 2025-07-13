from ogb.nodeproppred import NodePropPredDataset
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.data import Data

add_safe_globals({'torch_geometric.data.data.Data': Data})
from torch_geometric.data.data import DataEdgeAttr
add_safe_globals({'torch_geometric.data.data.DataEdgeAttr': DataEdgeAttr})

import torch
from torch_geometric.data import Data
from ogb.nodeproppred import NodePropPredDataset


dataset = NodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()

graph, labels = dataset[0]