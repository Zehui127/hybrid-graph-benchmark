from torch_geometric.datasets import Planetoid
from ..hg_formatter.HybridGraphFormatter import GraphFormatter
import os.path as osp
from typing import List, Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch_geometric.utils as utils


class OBG:
    def __new__(self,  name: str, root="/home/zl6222/repositories/data/", normalise=False,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name
        assert self.name in ['arxiv','products','proteins'], "Dataset not supported"
        path = osp.join(root,f"{self.name}",f"{self.name}_ordered.pt")
        # path = osp.join(root,f"{self.name}",f"{self.name}_ordered_adj.pt")
        data = torch.load(path)
        # optionally normalize the data
        split = {"train":data.train_mask,"val":data.val_mask,"test":data.test_mask}
        for key, idx in split.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask
        if normalise:
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t
        return [data]
