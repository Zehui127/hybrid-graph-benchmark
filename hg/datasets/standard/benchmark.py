from torch_geometric.datasets import Planetoid
from ..hg_formatter.HybridGraphFormatter import GraphFormatter
class Benchmark:
    """Try to directly construct the hypergraph from a original embedding"""
    def __new__(self,root,name,k=None,percentile=80):
        assert name in ["Cora", "CiteSeer", "PubMed"], "Dataset not supported"
        dataset = Planetoid(root, name)
        data = GraphFormatter(k,percentile)(dataset[0])
        return [data]
