import os.path as osp
from typing import List, Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data



class Grand(InMemoryDataset):
    r"""The Grand dataset
    Args:
        root (str): Root directory where the dataset should be saved.
        num_graphs (int): The number of graphs to be loaded.
     **STATS:**

    .. list-table::
        :widths: 10 10 10
        :header-rows: 1

        * - #Graphs
          - #features
          - #classes
        * - 60
          - 340
          - 3
        """
    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"

    file_id = {
        'Artery_Aorta': '1XjTRe4DJ2RU8SlqHkgUa6NTOdMLkGsd_',
        'Breast': '1cinfmlomxkenscWZwTCaXy8pyGdx78gs',
        'Vagina': '11tjAEuoeqDUfHVXt4cAIX_QPAOG-OD96',
        'Artery_Coronary': '1LyvWdPGXMelvVmh2g_RS3iRnUgBIJNf-',
        'Colon_adenocarcinoma': '1Ylngt1p_JgTKmP9GgyxZAA6eA1N4ST9c',
        'Sarcoma': '1SBzDq_WeYQxEW1c9mYphjOkB0Zts6TYT',
        'Liver': '1nbfLgOMEKqHy-o_XTvgqQ1p3s1X7uOtn',
        'Tibial_Nerve': '1lHvtRo7IF9BUPGaJL_eBL0aQD7IE3wvZ',
        'Kidney_renal_papillary_cell_carcinoma': '1Hj5zZFOon3YW3MTR0vzDrfS0IqgkwcNH',
        'Spleen': '1FRIb0moyq-L7eP5zUxINLmnZfqQ2qTDD',
    }
    # a help function used to evaluate the property of graphs.
    def split_rows(tensor):
        # Task 1: Split the first row into groups based on the second row
        unique_second_row_values = torch.unique(tensor[1])
        first_row_groups = {value.item(): [] for value in unique_second_row_values}
        for i, value in enumerate(tensor[1]):
            first_row_groups[value.item()].append(tensor[0][i].item())
        # Task 2: Split the second row into groups based on the first row
        unique_first_row_values = torch.unique(tensor[0])
        second_row_groups = {value.item(): [] for value in unique_first_row_values}
        for i, value in enumerate(tensor[0]):
            second_row_groups[value.item()].append(tensor[1][i].item())
        print(f"First row groups: {len(first_row_groups)}")
        print(f"Second row groups: {len(second_row_groups)}")
        return first_row_groups, second_row_groups
    def __init__(self,  root: str, name: str,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name
        assert self.name in ['Artery_Aorta','Breast','Vagina','Artery_Coronary','Colon_adenocarcinoma',
                             'Sarcoma','Liver','Tibial_Nerve','Kidney_renal_papillary_cell_carcinoma',
                             'Spleen']
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
        num_hyperedges = torch.unique(raw_data.hyperedge_index[1])
        data = Data(x=raw_data.x, edge_index=raw_data.adj, y=raw_data.y,
                    hyperedge_index=raw_data.hyperedge_index, num_hyperedges=len(num_hyperedges))

        torch.save(self.collate([data]), self.processed_paths[0])
