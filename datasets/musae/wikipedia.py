import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class Wikipedia(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent mutual hyperlinks between them. Hyperedges are mutually linked
    page groups that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3).
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average monthly traffic of the web page.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"chameleon"`, :obj:`"crocodile"`, :obj:`"squirrel"`).
        use_musae_node_embeddings (bool): If set to :obj:`True`, will load the pre-processed node embeddings
            as introduced in the `"Multi-scale Attributed Node Embedding"<https://arxiv.org/abs/1909.13021>`_ paper.
            If set to :obj:`False`, will load the raw node features extracted based on informative nouns appeared in
            the text of the Wikipedia articles. (default: :obj:`True`)
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

        * - Name
          - #nodes
          - #edges
          - #hyperedges
          - #features
        * - chameleon
          - 2,277
          - 62,742
          - 14,650
          - 128 or 3,132
        * - crocodile
          - 11,631
          - 341,546
          - 121,431
          - 128 or 13,183
        * - squirrel
          - 5,201
          - 396,706
          - 220,678
          - 128 or 3,148
    """
    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"
    file_id = {
        'chameleon': '1qLV8dMIhw-pe4ym16moOS9rl5rLD0FQr',
        'crocodile': '1D9c4D-V4iIxjklfaACGilrw_H0-rGVU5',
        'squirrel': '1-SBN-KSgSTpVJibG65slJDsBCBzXBj18',
    }

    NUM_HYPEREDGES = {
        'chameleon': 14650,
        'crocodile': 121431,
        'squirrel': 220678,
    }

    def __init__(self, root: str, name: str, use_musae_node_embeddings: bool = True,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.use_musae_node_embeddings = use_musae_node_embeddings
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
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
        return f'wikipedia_{self.name}.npz'

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
        y = torch.from_numpy(data['target']).to(torch.float)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        hyperedge_index = torch.from_numpy(data['hyperedges']).to(torch.long)

        data = Data(x=x, y=y, edge_index=edge_index, hyperedge_index=hyperedge_index,
                    num_hyperedges=self.NUM_HYPEREDGES[self.name])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
