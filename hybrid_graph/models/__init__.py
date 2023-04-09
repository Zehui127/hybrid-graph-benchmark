from .gnn.toynet import ToyNet
from .gnn.baseline import GCNNet, SAGENet


factory = {
    'toynet': ToyNet,
    'gcn': GCNNet,
    'sage': SAGENet,
}
