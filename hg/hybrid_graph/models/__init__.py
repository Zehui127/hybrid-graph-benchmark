from .gnn.baseline import GCNNet, SAGENet
from .gnn.gat import GATNet, GATV2Net
from .gnn.hyper import HyperGCN, HyperGAT
from .gnn.linearprobe import LPGCNHyperGCN,LPGGATGCN, LPGATHyperGCN, LPHYPERHYPER, LPGCNGCN, LPGGATGAT
from .ensemble.average_prediction import Average_Ensemble

factory = {
    'gcn': GCNNet,
    'sage': SAGENet,
    'gat': GATNet,
    'gatv2': GATV2Net,
    'hyper-gcn': HyperGCN,
    'hyper-gat': HyperGAT,
    'ensemble': Average_Ensemble,
    'lp-gcn-hyper-gcn': LPGCNHyperGCN,
    'lp-gat-hyper-gcn': LPGATHyperGCN,
    'lp-gat-gcn': LPGGATGCN,
    'lp-gcn-gcn': LPGCNGCN,
    'lp-gat-gat': LPGGATGAT,
    'lp-hyper-hyper': LPHYPERHYPER,
}
