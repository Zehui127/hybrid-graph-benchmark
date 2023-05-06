from torch_geometric.datasets import Planetoid
from .HybridGraphFormatter import GraphFormatter
import torch
class Benchmark:
    """Try to directly construct the hypergraph from a original embedding"""
    def __new__(self,root,name,k=None,percentile=80):
        assert name in ["Cora", "CiteSeer", "PubMed","ogbn-arxiv"], "Dataset not supported"
        if name == ogbn-arxiv:
            return [torch.load(root)]
        dataset = Planetoid(root, name)
        data = GraphFormatter(k,percentile)(dataset[0])
        return [data]
