from .gnn.toynet import ToyNet
from .gnn.baseline import GCNNet, SAGENet
from .gnn.hybrid import HybridGCN, HybridSAGE

factory = {
    'toynet': ToyNet,
    'gcn': GCNNet,
    'sage': SAGENet,
    'hybrid-gcn': HybridGCN,
    'hybrid-sage': HybridSAGE,
}
