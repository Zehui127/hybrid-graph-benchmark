from .gnn.toynet import ToyNet
from .gnn.baseline import GCNNet, SAGENet
from .gnn.hybrid import HybridGCN, HybridSAGE
from .gnn.gat import GATNet, GATV2Net
from .gnn.hyper import HyperGCN, HyperGAT
from .ensemble.average_prediction import Average_Ensemble

factory = {
    'toynet': ToyNet,
    'gcn': GCNNet,
    'sage': SAGENet,
    'gat': GATNet,
    'gatv2': GATV2Net,
    'hybrid-gcn': HybridGCN,
    'hybrid-sage': HybridSAGE,
    'hyper-gcn': HyperGCN,
    'hyper-gat': HyperGAT,
    'ensemble': Average_Ensemble,
}
