import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import trimesh
mesh=trimesh.load("test.obj")
vertices=mesh.vertices
edges=mesh.edges
faces=mesh.faces


vertices_tensor=torch.tensor(vertices,dtype=torch.float)
edges_tensor=torch.tensor(edges,dtype=torch.long).t().contiguous()  # t是转置操作，contiguous是连续存储

data=Data(x=vertices_tensor,edge_index=edges_tensor)

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
G=to_networkx(data)
nx.draw(G)
plt.show()