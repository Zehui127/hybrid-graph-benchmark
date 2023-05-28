import os.path as osp
from typing import List, Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch_geometric.utils as utils


class Grand(InMemoryDataset):
    r"""The Grand dataset
    Args:
        root (str): Root directory where the dataset should be saved.
        num_graphs (int): The number of graphs to be loaded.
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
        * - Leukemia
          - 4,651
          - 6,362
          - 7,812
          - 4608
          - 3
        * - Brain
          - 6,196
          - 6,245
          - 11,878
          - 4608
          - 3
        * - Kidney_renal_papillary_cell_carcinoma
          - 4,319
          - 5,599
          - 7,369
          - 4608
          - 3
        * - Lung
          - 6,119
          - 6,160
          - 11,760
          - 4608
          - 3
        * - Breast
          - 5,921
          - 5,910
          - 11,400
          - 4608
          - 3
        * - Artery_Coronary
          - 5,755
          - 5,722
          - 11,222
          - 4608
          - 3
        * - Artery_Aorta
          - 5,848
          - 5,823
          - 11,368
          - 4608
          - 3
        * - Lung_cancer
          - 4,896
          - 6,995
          - 8,179
          - 4608
          - 3
        * - Stomach_cancer
          - 4,518
          - 6,051
          - 7,611
          - 4608
          - 3
        * - Stomach
          - 5,745
          - 5,694
          - 11,201
          - 4608
          - 3
        """
    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"
    file_id = {
        'Leukemia': '1ztRJcNXG4O6OY106JFyZ8MF1EyV55n1_',
        'Brain': '1kBxVOj_H0FYRc159HypgvLw5zwsp-9Pv',
        'Kidney_renal_papillary_cell_carcinoma': '1Hj5zZFOon3YW3MTR0vzDrfS0IqgkwcNH',
        'Lung': '14g1wbCRbepZaniRdsCD-TP0p6SPTmvhS',
        'Breast': '1cinfmlomxkenscWZwTCaXy8pyGdx78gs',
        'Artery_Coronary': '1LyvWdPGXMelvVmh2g_RS3iRnUgBIJNf-',
        'Artery_Aorta': '1XjTRe4DJ2RU8SlqHkgUa6NTOdMLkGsd_',
        'Lung_cancer': '1QLRHsyhssELyC4eJ9sH75E3QJ5lHUNrA',
        'Stomach_cancer':'1TF0rg3agGfI9ScZijcQgH6FhkZ6O6EsB',
        'Stomach':'1MWc_wINEogQHX5u6xvvOWyohtd_HIwoj',
    }
    def __init__(self,  root: str, name: str,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name
        assert self.name in ['Artery_Aorta','Breast','Artery_Coronary','Stomach_cancer',
                             'Stomach', 'Brain','Lung',
                             'Kidney_renal_papillary_cell_carcinoma',
                             'Lung_cancer','Leukemia']
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
        data = Data(x=raw_data.x, edge_index=raw_data.edge_index, y=raw_data.y,
                    hyperedge_index=raw_data.hyperedge_index, num_hyperedges=raw_data.num_hyperedges)

        torch.save(self.collate([data]), self.processed_paths[0])

    def to_undirected(self, edge_index):
        # Given an edge_index tensor of shape [2, E], we first create a new tensor
        # that includes both the original edges and their reverse.
        undirected_edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)

        # In the new tensor, there may be duplicate edges, so we call unique to remove them.
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1)

        return undirected_edge_index
