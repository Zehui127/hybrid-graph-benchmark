from torch_geometric.datasets import Planetoid
from ..hg_formatter.HybridGraphFormatter import GraphFormatter
import os.path as osp
from typing import List, Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch_geometric.utils as utils


class Benchmark(InMemoryDataset):
    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"
    file_id = {
        'Pubmed': '1w4E2sC5yVyz0Gn10ndTPMuZzx2hDgYLo',
        'CoraAuthor': '1Agq42w-geVR-o-lWHKRdo8lA6Qdf-mwl',
        'CoraCite': '1Dq2pKiOeW_bTYCAwJJJznf1HXzkMEJ_5',
    }
    def __init__(self,  root: str, name: str,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name
        assert self.name in ['Pubmed','CoraAuthor','CoraCite']
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    @property
    def raw_file_names(self) -> List[str]:
        return f'{self.name}.pt'
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.file_id[self.name]), self.raw_dir, filename=self.raw_file_names)

    def process(self):
        print("Processing the data...")
        raw_data = torch.load(osp.join(self.raw_dir, self.raw_file_names))
        # torch.arange(30171).view(-1,1).float()
        num_hyperedges = torch.unique(raw_data.hyperedge_index[1,:])

        data = Data(x=raw_data.x, edge_index=raw_data.edge_index, y=raw_data.y,
                    hyperedge_index=raw_data.hyperedge_index, num_hyperedges=num_hyperedges)

        torch.save(self.collate([data]), self.processed_paths[0])

    def to_undirected(self, edge_index):
        # Given an edge_index tensor of shape [2, E], we first create a new tensor
        # that includes both the original edges and their reverse.
        undirected_edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)

        # In the new tensor, there may be duplicate edges, so we call unique to remove them.
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1)

        return undirected_edge_index
