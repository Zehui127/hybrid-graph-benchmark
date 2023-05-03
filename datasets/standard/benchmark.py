from torch_geometric.datasets import Planetoid
from .HybridGraphFormatter import GraphFormatter

class Benchmark:
    def __new__(self,root,name,k=3,percentile=90):
        assert name in ["Cora", "CiteSeer", "PubMed"], "Dataset not supported"
        dataset = Planetoid(root, name)
        data = GraphFormatter(k,percentile)(dataset[0])
        return data
