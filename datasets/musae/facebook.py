import os
from typing import Callable, Optional, List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class Facebook(InMemoryDataset):
    r"""The Facebook page-page network dataset introduced in the
    `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent verified pages on Facebook and edges are mutual likes. Hyperedges are mutually liked page groups
    that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3).
    It contains 22,470 nodes, 342,004 edges, 236,663 hyperedges, 128 (if the MUSAE preprocessed node embeddings
    are used) or 4,714 (if the raw node features are used) node features, and 4 classes.

    Args:
        root (str): Root directory where the dataset should be saved.
        use_musae_node_embeddings (bool): If set to :obj:`True`, will load the pre-processed node embeddings
            as introduced in the `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
            If set to :obj:`False`, will load the raw node features extracted from the site descriptions.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data`
            object and returns a transformed version. The data object will be transformed before being saved to disk.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 15 15 15 15
        :header-rows: 1

        * - #nodes
          - #edges
          - #hyperedges
          - #features
          - #classes
        * - 22,470
          - 342,004
          - 236,663
          - 128 or 4,714
          - 4
    """

    url_data = 'https://graphmining.ai/datasets/ptg/facebook.npz'
    url_preprocessed = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1uOw9eR_ax8REizC4sghrvBT7sVAZph5_'

    NUM_HYPEREDGES = 236663

    def __init__(self, root: str, use_musae_node_embeddings: bool = True,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.use_musae_node_embeddings = use_musae_node_embeddings
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        if self.use_musae_node_embeddings:
            return os.path.join(self.root, 'processed', 'musae_node_embeddings')
        else:
            return os.path.join(self.root, 'processed', 'raw_node_features')

    @property
    def raw_file_names(self) -> List[str]:
        return ['facebook.npz', 'facebook_preprocessed.npz']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url_data, self.raw_dir)
        download_url(self.url_preprocessed, self.raw_dir, filename=self.raw_file_names[1])

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        preprocessed = np.load(self.raw_paths[1], 'r', allow_pickle=True)
        if self.use_musae_node_embeddings:
            x = torch.from_numpy(data['features']).to(torch.float)
        else:
            x = torch.from_numpy(preprocessed['raw_features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()
        hyperedge_index = torch.from_numpy(preprocessed['hyperedges']).to(torch.long)

        data = Data(x=x, y=y, edge_index=edge_index, hyperedge_index=hyperedge_index,
                    num_hyperedges=self.NUM_HYPEREDGES)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
