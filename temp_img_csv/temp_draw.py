import networkx as nx
import torch_geometric

import datasets
from datasets import grand,musae
from torch_geometric.utils import degree
import pandas as pd
import torch

dataset = grand.Grand("data/grand","Artery_Aorta")
# dataset = musae.Twitch("data/musae",'DE')
data = dataset[0]
print(data)
data.num_nodes = data.x.shape[0]
degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

# Compute the average degree
average_degree = torch.mean(degrees)
print(f"Average degree: {average_degree:.2f}")

g = torch_geometric.utils.to_networkx(data, to_undirected=True)
edges_df = pd.DataFrame(g.edges(), columns=["source", "target"])
edges_df.to_csv("edges_grand.csv", index=False)
