import os.path as osp
from typing import List
import torch
from torch_geometric.data import InMemoryDataset, download_url
import tarfile
import torch_geometric.utils as utils


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
    num_graphs = 10
    url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"

    def __init__(self, name: str, root='cache'):
        super().__init__(root)
        self.name = name
        self.data = torch.load(self.processed_paths[0])[self.get_index_name]
    @property
    def get_index_name(self):
        indices = [index for index, element in enumerate(self.raw_file_names) if element == f"{self.name}.pt"]
        return indices[0]
    @property
    def raw_file_names(self) -> List[str]:
        return ['Artery_Aorta.pt', 'Breast.pt', 'Vagina.pt',
                'Artery_Coronary.pt',
                'Colon_adenocarcinoma.pt', 'Sarcoma.pt', 'Liver.pt',
                'Tibial_Nerve.pt', 'Kidney_renal_papillary_cell_carcinoma.pt',
                'Spleen.pt', 'Esophagus_Muscularis.pt', 'Adrenal_Gland.pt',
                'Stomach_adenocarcinoma.pt', 'Pituitary.pt', 'Ovary.pt',
                'Esophageal_carcinoma.pt', 'Lung_squamous_cell_carcinoma.pt',
                'Thymoma.pt', 'Lung_adenocarcinoma.pt',
                'Brain_Lower_Grade_Glioma.pt',
                'Uveal_Melanoma.pt', 'Testis.pt', 'Kidney_Chromophobe.pt',
                'Colon_Sigmoid.pt', 'Skin_Cutaneous_Melanoma.pt',
                'Pancreas.pt',
                'Pheochromocytoma_and_Paraganglioma.pt',
                'Brain_Basal_Ganglia.pt',
                'Thyroid.pt', 'Skin.pt',
                'Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma.pt',
                'Heart_Atrial_Appendage.pt', 'Prostate.pt', 'Kidney_Cortex.pt',
                'Stomach.pt', 'Adipose_Visceral.pt', 'Whole_Blood.pt',
                'Adrenocortical_carcinoma.pt',
                'Gastroesophageal_Junction.pt', 'Lung.pt', 'Uterus.pt',
                'Artery_Tibial.pt', 'Brain_Cerebellum.pt',
                'Thyroid_carcinoma.pt',
                'Kidney_renal_clear_cell_carcinoma.pt',
                'Intestine_Terminal_Ileum.pt',
                'Esophagus_Mucosa.pt', 'Glioblastoma_multiforme.pt',
                'Cholangiocarcinoma.pt',
                'Minor_Salivary_Gland.pt', 'Colon_Transverse.pt',
                'Bladder_Urothelial_Carcinoma.pt',
                'Skeletal_Muscle.pt', 'Adipose_Subcutaneous.pt',
                'Head_and_Neck_squamous_cell_carcinoma.pt',
                'Acute_Myeloid_Leukemia.pt',
                'Heart_Left_Ventricle.pt', 'Rectum_adenocarcinoma.pt',
                'Brain_Other.pt', 'Mesothelioma.pt'][:self.num_graphs]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        if self.num_graphs == 10:
            self.url = self.url.format("1joouDSH_XURrj18JwAazFyJx3Ta4ZTPG")
        else:
            self.url = self.url.format("1u9B4zBALZ56naWistLfDPxvHlGIJHg0c")
        # Download the file
        path = download_url(self.url, self.raw_dir)
        # Untar the folder
        print("Untar the datasets...")
        with tarfile.open(path, "r") as tar:
            # Extract all contents of the tar file to the output directory
            tar.extractall(self.raw_dir)

    def process(self):
        data_list = []
        print("Processing the data...")
        for path in self.raw_file_names:
            print(f"Graph Name: {path}")
            data = torch.load(osp.join(self.raw_dir, path))
            data.edge_index = data.adj
            data = self.random_node_split(data)
            data = self.random_edge_split(data)
            data_list.append(data)
        torch.save(data_list, self.processed_paths[0])

    def random_node_split(self, data, train_ratio=0.6, val_ratio=0.2):
        num_nodes = data.y.shape[0]
        num_train = int(train_ratio * num_nodes)
        num_val = int(val_ratio * num_nodes)

        perm = torch.randperm(num_nodes)
        train_mask = perm[:num_train]
        val_mask = perm[num_train:num_train + num_val]
        test_mask = perm[num_train + num_val:]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_mask] = True
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[val_mask] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_mask] = True
        return data

    def random_edge_split(self, data, train_ratio=0.6, val_ratio=0.2):
        # Convert to undirected graph
        edge_index = utils.to_undirected(data.edge_index)

        num_edges = edge_index.size(1)
        num_train = int(train_ratio * num_edges)
        num_val = int(val_ratio * num_edges)

        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[:num_train]]
        val_edges = edge_index[:, perm[num_train:num_train + num_val]]
        test_edges = edge_index[:, perm[num_train + num_val:]]

        data.train_pos_edge_index = train_edges
        data.val_pos_edge_index = val_edges
        data.test_pos_edge_index = test_edges

        return data
