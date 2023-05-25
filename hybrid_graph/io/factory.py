import os
import sys
import math
import signal
import functools
import numpy as np

import torch
import torch_geometric
from torch_geometric import transforms
import torch_geometric.utils as utils
import logging

from .utils import device
import pathlib

torch.multiprocessing.set_sharing_strategy('file_system')

# Add the dataset directory to sys.path
dataset_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), 'datasets')
if dataset_dir not in sys.path:
    sys.path.append(dataset_dir)

import datasets

DATASET_INFO = {
    'grand_ArteryAorta': {
        'type': 'Grand',
        'name': 'Artery_Aorta',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,
        }

    },
    'grand_Cholang': {
        'type': 'Grand',
        'name': 'Cholang',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,
        }

    },
    'grand_ArteryCoronary': {
        'type': 'Grand',
        'name': 'Artery_Coronary',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'grand_Breast': {
        'type': 'Grand',
        'name': 'Breast',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

        }

    },
    'grand_Brain': {
        'type': 'Grand',
        'name': 'Brain',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

        }

    },
   'grand_Lung': {
        'type': 'Grand',
        'name': 'Lung',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

               }
    },
   'grand_Stomach': {
        'type': 'Grand',
        'name': 'Stomach',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

               }
    },
    'grand_Leukemia': {
        'type': 'Grand',
        'name': 'Leukemia',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'grand_Lungcancer': {
        'type': 'Grand',
        'name': 'Lung_cancer',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

               }
    },
    'grand_Stomachcancer': {
        'type': 'Grand',
        'name': 'Stomach_cancer',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

               }
    },
    'grand_KidneyCancer': {
        'type': 'Grand',
        'name': 'Kidney_renal_papillary_cell_carcinoma',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 4608,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'grand_Vagina': {
        'type': 'Grand',
        'name': 'Vagina',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': True,

        }
    },
    'grand_Sarcoma': {
        'type': 'Grand',
        'name': 'Sarcoma',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': True,

        }
    },
    'grand_Liver': {
        'type': 'Grand',
        'name': 'Liver',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': True,

        }
    },
    'grand_TibialNerve': {
        'type': 'Grand',
        'name': 'Tibial_Nerve',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': True,

        }
    },
    'grand_Spleen': {
        'type': 'Grand',
        'name': 'Spleen',
        'root': 'data/grand',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': True,
               }
    },
    'musae_Twitch_DE': {
        'type': 'Twitch',
        'name': 'DE',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Twitch_EN': {
        'type': 'Twitch',
        'name': 'EN',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Twitch_ES': {
        'type': 'Twitch',
        'name': 'ES',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Twitch_FR': {
        'type': 'Twitch',
        'name': 'FR',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Twitch_PT': {
        'type': 'Twitch',
        'name': 'PT',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Twitch_RU': {
        'type': 'Twitch',
        'name': 'RU',
        'root': 'data/musae',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Facebook': {
        'type': 'Facebook',
        'root': 'data/musae/facebook',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 4,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Github': {
        'type': 'GitHub',
        'root': 'data/musae/github',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 4,
            'is_regression': False,
            'is_edge_pred': False,

        }
    },
    'musae_Wiki_chameleon': {
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name': 'chameleon',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
            'is_edge_pred': False,

        }
    },
    'musae_Wiki_crocodile': {
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name': 'crocodile',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
            'is_edge_pred': False,

        }
    },
    'musae_Wiki_squirrel': {
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name': 'squirrel',
        'single_graph': True,
        'info': {
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
            'is_edge_pred': False,

        }
    },
    'benchmark_Cora': {
        'type': 'Benchmark',
        'name': 'Cora',
        'root': 'data/benchmark',
        'single_graph': True,
        'info': {
            'original_mask': True,
            'num_node_features': 1433,
            'num_classes': 7,
            'is_regression': False,
            'is_edge_pred': False,
        }
    },
    'benchmark_Citeseer': {
        'type': 'Benchmark',
        'name': 'CiteSeer',
        'root': 'data/benchmark',
        'single_graph': True,
        'info': {
            'original_mask': True,
            'num_node_features': 3703,
            'num_classes': 6,
            'is_regression': False,
            'is_edge_pred': False,
        }
    },
    'benchmark_PubMed': {
        'type': 'Benchmark',
        'name': 'PubMed',
        'root': 'data/benchmark',
        'single_graph': True,
        'info': {
            'original_mask': True,
            'num_node_features': 500,
            'num_classes': 3,
            'is_regression': False,
            'is_edge_pred': False,
        }
    },
    'benchmark_ogbn-arxiv': {
    'type': 'Benchmark',
    'name': 'ogbn-arxiv',
    'root': 'data/ogbn-arxiv.pt',
    'single_graph': True,
    'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 40,
            'is_regression': False,
            'is_edge_pred': False,
            }
    },

    'amazon_Photo':  {
        'type': 'Amazon',
        'name': 'Photos',
        'root': 'data/amazon',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 1000,
                'num_classes': 10,
                'is_regression': False,
                'is_edge_pred': False,
        }
    },
    'amazon_Computer':  {
        'type': 'Amazon',
        'name': 'Computers',
        'root': 'data/amazon',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 1000,
                'num_classes': 10,
                'is_regression': False,
                'is_edge_pred': False,
        }
    },
}


class DataLoader(torch_geometric.loader.DataLoader):
    pin_memory = True

    def __init__(
            self, dataset, masks, batch_size,
            workers, single_graph=False, shuffle=False, onehot=False, sampler=None, batch_sampler=None):
        super().__init__(
            dataset, batch_size,
            pin_memory=self.pin_memory,
            num_workers=workers, worker_init_fn=self.worker_init,
            shuffle=shuffle, sampler=sampler, batch_sampler=None)
        self.single_graph = single_graph
        self.masks = masks
        self.onehot = onehot
        self.workers = workers
        # self.sampler = sampler
        # TODO sampler is potentially needed
        # self.sampler = None
        # self.batch_sampler = None

    @staticmethod
    def worker_init(x):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # TODO implement hyper-graph partition
    def __iter__(self):
        if self.single_graph:
            for i, item in enumerate(super().__iter__()):
                yield item
        else:
            for i, item in enumerate(self.sampler.__iter__()):
                yield item


def get_dataset(name, original_mask=False, split=0.9, batch_size=6000, workers=2, num_steps=5):
    # fix random seeds
    np.random.seed(1)
    torch.manual_seed(1)

    info = dict(DATASET_INFO[name])
    dataset_info = info.pop('info', {})

    single_graph = info.pop('single_graph', False)
    onehot = info.pop('onehot', False)

    cls = getattr(datasets, info.pop('type'))
    print(info)
    dataset = cls(**info)
    kwargs = {
        'batch_size': 1,
        'workers': workers,
        'single_graph': single_graph,
    }


    original_mask = dataset_info.pop('original_mask')
    Loader = functools.partial(DataLoader, **kwargs)
    if dataset_info['is_edge_pred']:
        dataset = create_edge_label(dataset)
    dataset, masks = mask_split(dataset, original_mask)
    # take one sample mask out
    train_mask, eval_mask, test_mask = masks[0]
    dataset = dataset[0]
    dataset.train_mask = train_mask
    dataset.val_mask = eval_mask
    dataset.test_mask = test_mask
    print(dataset)
    # dataloader requires a list of dataset
    dataset = [dataset]
    # logging.info(
    print(
        f"Search with a partition of {train_mask.sum()} train data, "
        f"{eval_mask.sum()} val data and {test_mask.sum()} test data.")
    # for single graph the masks is of no use
    print(dataset_info)
    if single_graph:
        return Loader(dataset, masks), Loader(dataset, masks), Loader(dataset, masks), dataset_info
    else:
        kwargs = {
            "batch_size": batch_size,
            "num_steps": num_steps,
            "num_workers": workers,
        }
        Sampler = functools.partial(datasets.HypergraphSAINTNodeSampler, **kwargs)
        return (
            Loader(dataset, masks, sampler=Sampler(dataset[0])),
            Loader(dataset, masks, sampler=Sampler(dataset[0])),
            Loader(dataset, masks, sampler=Sampler(dataset[0])),
            dataset_info,
        )
def mask_split(dataset, original_mask=True,
               train_portion=0.8, eval_portion=0.1, test_portion=0.1):
    # re split to the train, eval and test mask to 60:20:20
    # only suppose to work with cora
    masks = []
    for data in dataset:
        if original_mask:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            num_nodes = data.num_nodes
            indexes = np.arange(num_nodes)
            np.random.shuffle(indexes)
            zeros = torch.zeros((num_nodes))

            train_point = math.ceil(num_nodes * train_portion)
            eval_point = math.ceil(num_nodes * (train_portion + eval_portion))

            train_mask = zeros.clone()
            val_mask = zeros.clone()
            test_mask = zeros.clone()
            train_mask[indexes[:train_point]] = 1
            val_mask[indexes[train_point:eval_point]] = 1
            test_mask[indexes[eval_point:]] = 1
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            train_mask = train_mask.type(torch.bool)
            val_mask = val_mask.type(torch.bool)
            test_mask = test_mask.type(torch.bool)

        masks.append((train_mask, val_mask, test_mask))
    return dataset, masks


def random_node_split(data, train_ratio=0.6, val_ratio=0.2, keep_joint_hyperedges=True):
    num_nodes = data.num_nodes
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.zeros(num_nodes)

    perm = torch.randperm(num_nodes)
    train_mask[perm[:num_train]] = 1
    val_mask[perm[num_train:num_train + num_val]] = 1
    test_mask[perm[num_train + num_val:]] = 1

    data.train_mask = train_mask.type(torch.bool)
    data.val_mask = val_mask.type(torch.bool)
    data.test_mask = test_mask.type(torch.bool)

    if keep_joint_hyperedges:
        # If keep_joint_hyperedges is set to True, then a hyperedge is included in a split dataset
        # as long as it contains a node in that split dataset.
        data.train_hyperedge_index = data.hyperedge_index[:, data.train_mask[data.hyperedge_index[0, :]]]
        data.val_hyperedge_index = data.hyperedge_index[:, data.val_mask[data.hyperedge_index[0, :]]]
        data.test_hyperedge_index = data.hyperedge_index[:, data.test_mask[data.hyperedge_index[0, :]]]
        data.num_train_hyperedges = data.train_hyperedge_index[1].unique().numel()
        data.num_val_hyperedges = data.val_hyperedge_index[1].unique().numel()
        data.num_test_hyperedges = data.test_hyperedge_index[1].unique().numel()
    else:
        # If keep_joint_hyperedges is set to False, then a hyperedge is included in a split dataset
        # only if every node in that hyperedge belong to that split dataset. This would drastically
        # reduce the number of total hyperedges in all splits.
        hyperedge_index = data.hyperedge_index
        train_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[:num_train])
        val_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[num_train:num_train + num_val])
        test_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[num_train + num_val:])

        train_hyperedge_to_exclude_mask = hyperedge_index[1, ~train_node_in_hyperedge_mask].unique()
        val_hyperedge_to_exclude_mask = hyperedge_index[1, ~val_node_in_hyperedge_mask].unique()
        test_hyperedge_to_exclude_mask = hyperedge_index[1, ~test_node_in_hyperedge_mask].unique()

        train_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], train_hyperedge_to_exclude_mask)
        val_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], val_hyperedge_to_exclude_mask)
        test_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], test_hyperedge_to_exclude_mask)

        data.train_hyperedge_index = hyperedge_index[:, train_hyperedge_mask]
        data.val_hyperedge_index = hyperedge_index[:, val_hyperedge_mask]
        data.test_hyperedge_index = hyperedge_index[:, test_hyperedge_mask]
        data.num_train_hyperedges = data.num_hyperedges - train_hyperedge_to_exclude_mask.numel()
        data.num_val_hyperedges = data.num_hyperedges - val_hyperedge_to_exclude_mask.numel()
        data.num_test_hyperedges = data.num_hyperedges - test_hyperedge_to_exclude_mask.numel()

    return data


def random_hyperedge_split(data, train_ratio=0.6, val_ratio=0.2, keep_joint_hyperedges=True):
    # Split the dataset by sampling hyperedges and putting all nodes
    # within the sampled hyperedges in the same split dataset.
    # WARNING: THIS ONLY WORKS IN THEORY. DO NOT USE.
    num_nodes = data.num_nodes
    num_hyperedges = data.num_hyperedges
    hyperedge_index = data.hyperedge_index

    num_train_expected = int(train_ratio * num_nodes)
    num_val_expected = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.ones(num_nodes)
    train_hyperedge_mask = torch.zeros(num_hyperedges)
    val_hyperedge_mask = torch.zeros(num_hyperedges)
    test_hyperedge_mask = torch.zeros(num_hyperedges)
    train_joint_hyperedge_mask = torch.zeros(num_hyperedges)
    val_joint_hyperedge_mask = torch.zeros(num_hyperedges)
    test_joint_hyperedge_mask = torch.zeros(num_hyperedges)

    # Sample hyperedges into the training set until it contains at least num_train_expected nodes
    perm = torch.randperm(num_hyperedges)
    i = 0
    while int(torch.sum(train_mask)) <= num_train_expected:
        sample = perm[i]
        train_hyperedge_mask[sample] = 1
        train_mask[hyperedge_index[0, hyperedge_index[1, :] == sample]] = 1
        i += 1

    # Remove hyperedges containing nodes in the traning set from the remaining hyperedges
    # WARNING: THIS MAY NOT LEAVE ENOUGH HYPEREDGES FOR THE REMAINING SPLITS, OR EVEN
    # REMOVE ALL REMAINING HYPEREDGES.
    perm = perm[i:]
    train_joint_hyperedges = hyperedge_index[1, train_mask[hyperedge_index[0, :]] == 1].unique()
    train_joint_hyperedge_mask[train_joint_hyperedges] = 1
    perm = perm[~torch.isin(perm, train_joint_hyperedges)]
    assert perm.numel() > 0

    # Sample hyperedges into the validation set until the combination of training and validation sets
    # contain at least num_train_expected + num_val_expected nodes, or all remaining hyperedges
    # have been sampled.
    num_train_actual = int(torch.sum(train_mask))
    i = 0
    while num_train_actual + int(torch.sum(val_mask)) <= num_train_expected + num_val_expected \
            or i == perm.numel():
        sample = perm[i]
        val_hyperedge_mask[sample] = 1
        val_mask[hyperedge_index[1, :] == sample] = 1
        i += 1

    # Remove hyperedges containing nodes in the validation set from the remaining hyperedges
    # WARNING: THIS MAY NOT LEAVE ENOUGH HYPEREDGES FOR THE TEST SET, OR EVEN
    # REMOVE ALL REMAINING HYPEREDGES.
    perm = perm[i:]
    assert perm.numel() > 0
    val_joint_hyperedges = hyperedge_index[1, val_mask[hyperedge_index[0, :]] == 1].unique()
    val_joint_hyperedge_mask[val_joint_hyperedges] = 1
    perm = perm[~torch.isin(perm, val_joint_hyperedges)]
    assert perm.numel() > 0

    # Put all remaining nodes into the test set, and find corresponding hyperedges
    test_mask[train_mask == 1] = 0
    test_mask[val_mask == 1] = 0
    test_joint_hyperedges = hyperedge_index[1, test_mask[hyperedge_index[0, :]] == 1].unique()
    test_joint_hyperedge_mask[test_joint_hyperedges] = 1
    test_hyperedge_mask[test_joint_hyperedges] = 1
    test_hyperedge_mask[train_hyperedge_mask == 1] = 0
    test_hyperedge_mask[val_hyperedge_mask == 1] = 0

    # output masks
    data.train_mask = train_mask.type(torch.bool)
    data.val_mask = val_mask.type(torch.bool)
    data.test_mask = test_mask.type(torch.bool)
    if keep_joint_hyperedges:
        data.train_hyperedge_mask = train_joint_hyperedge_mask.type(torch.bool)
        data.val_hyperedge_mask = val_joint_hyperedge_mask.type(torch.bool)
        data.test_hyperedge_mask = test_joint_hyperedge_mask.type(torch.bool)
    else:
        data.train_hyperedge_mask = train_hyperedge_mask.type(torch.bool)
        data.val_hyperedge_mask = val_hyperedge_mask.type(torch.bool)
        data.test_hyperedge_mask = test_hyperedge_mask.type(torch.bool)
    return data


def random_edge_split(data, train_ratio=0.6, val_ratio=0.2):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        num_train = int(train_ratio * num_edges)
        num_val = int(val_ratio * num_edges)
        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[:num_train]]
        val_edges = edge_index[:, perm[num_train:num_train + num_val]]
        test_edges = edge_index[:, perm[num_train + num_val:]]
        return train_edges, val_edges, test_edges

def create_edge_label(datasets,train_ratio=0.6, val_ratio=0.2):
        r"""This is used for edge prediction tasks
        Creates edge labels :obj:`data.edge_label` based on node labels
        :obj:`data.y`.

        Args:
            data (torch_geometric.data.Data): The graph data object.

        :rtype: :class:`torch_geometric.data.Data`
        """
        def edge_label(data):
            # we split the edge_index with all mode
            # get the edge index for train_message_ps = eval_mp,
            # train_supervision + eval_supervision + test_supervision = whole graph
            # test_mp = train_supervision + eval_supervision
            # data.edge_index = utils.to_undirected(data.edge_index)
            train_pos_edge_index, val_pos_edge_index, test_pos_edge_index  = random_edge_split(data,train_ratio, val_ratio)
            data.train_edge_index = data.val_edge_index = train_pos_edge_index
            data.test_edge_index = torch.cat([train_pos_edge_index, val_pos_edge_index], dim=1)
            data.train_label, data.train_edge_label_index = helper(train_pos_edge_index)
            data.val_label, data.val_edge_label_index = helper(val_pos_edge_index)
            data.test_label, data.test_edge_label_index = helper(test_pos_edge_index)
            return data
        def helper(edge_index):
            neg_edge_index = utils.negative_sampling(edge_index)
            # create edge_label
            pos_label = torch.ones(edge_index.size(1), dtype=torch.int64)
            neg_label = torch.zeros(neg_edge_index.size(1), dtype=torch.int64)
            edge_label = torch.cat([pos_label, neg_label], dim=0)
            return edge_label, torch.cat([edge_index, neg_edge_index], dim=1)
        return [edge_label(data) for data in datasets]
