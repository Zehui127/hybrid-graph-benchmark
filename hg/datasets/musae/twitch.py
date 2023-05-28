import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class Twitch(InMemoryDataset):
    r"""The Twitch Gamer networks introduced in the
    `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent gamers on Twitch and edges are followerships between them. Hyperedges are mutually following
    user groups that contain at least 3 gamers (i.e., maximal cliques with sizes of at least 3).
    The task is to predict whether a user streams mature content.
    Each dataset contains 128 (if the MUSAE preprocessed node embeddings are used) or 4,005 (if the raw node features
    are used) node features, and 2 classes.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset
            (:obj:`"DE"`, :obj:`"EN"`, :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
        use_musae_node_embeddings (bool): If set to :obj:`True`, will load the pre-processed node embeddings
            as introduced in the `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
            If set to :obj:`False`, will load the raw node features extracted based on the games played and liked,
            location and streaming habits. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data`
            object and returns a transformed version. The data object will be transformed before being saved to disk.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 15 15 15 15 15
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #hyperedges
          - #features
          - #classes
        * - DE
          - 9,498
          - 306,276
          - 297,315
          - 128 or 3,170
          - 2
        * - EN
          - 7,126
          - 70,648
          - 13,248
          - 128 or 3,170
          - 2
        * - ES
          - 4,648
          - 118,764
          - 77,135
          - 128 or 3,170
          - 2
        * - FR
          - 6,549
          - 225,332
          - 172,653
          - 128 or 3,170
          - 2
        * - PT
          - 1,912
          - 62,598
          - 74,830
          - 128 or 3,170
          - 2
        * - RU
          - 4,385
          - 74,608
          - 25,673
          - 128 or 3,170
          - 2
    """

    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"
    file_id = {
        'DE': '148A6DhddSz2qeWnlHwyTBJl8eLkhoPzv',
        'EN': '1vb8KrwvZvBHdT_dll6oe62HNk0EQfSOb',
        'ES': '1u-AXJ6En7ctBhPbwbW1dcCf946MmJWlS',
        'FR': '1WLpZmpQPhFIrnsHQ-4-8jxkYnorUIYS8',
        'PT': '13gVESn3AGbyoF8aveWrJeAPqUGwTG09o',
        'RU': '137tr2fJt-lKaQ_PcMSCl3gh33Gateke_'
    }

    NUM_HYPEREDGES = {
        'DE': 297315,
        'EN': 13248,
        'ES': 77135,
        'FR': 172653,
        'PT': 74830,
        'RU': 25673
    }

    def __init__(self, root: str, name: str, use_musae_node_embeddings: bool = True,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name.upper()
        self.use_musae_node_embeddings = use_musae_node_embeddings
        assert self.name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.use_musae_node_embeddings:
            return os.path.join(self.root, self.name, 'processed', 'musae_node_embeddings')
        else:
            return os.path.join(self.root, self.name, 'processed', 'raw_node_features')

    @property
    def raw_file_names(self) -> str:
        return f'twitch_{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.file_id[self.name]), self.raw_dir, filename=self.raw_file_names)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        if self.use_musae_node_embeddings:
            x = torch.from_numpy(data['features']).to(torch.float)
        else:
            x = torch.from_numpy(data['raw_features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        hyperedge_index = torch.from_numpy(data['hyperedges']).to(torch.long)

        data = Data(x=x, y=y, edge_index=edge_index, hyperedge_index=hyperedge_index,
                    num_hyperedges=self.NUM_HYPEREDGES[self.name])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
