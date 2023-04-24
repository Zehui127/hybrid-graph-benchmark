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
dataset_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(),'datasets')
if dataset_dir not in sys.path:
    sys.path.append(dataset_dir)

import datasets

DATASET_INFO = {
    'grand_ArteryAorta': {
        'type': 'Grand',
        'name': 'Artery_Aorta',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }

    },
    'grand_Breast': {
        'type': 'Grand',
        'name': 'Breast',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }

    },
    'grand_Vagina': {
        'type': 'Grand',
        'name': 'Vagina',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_ArteryCoronary': {
        'type': 'Grand',
        'name': 'Artery_Coronary',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_ColonAdenocarcinoma': {
        'type': 'Grand',
        'name': 'Colon_adenocarcinoma',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_Sarcoma': {
        'type': 'Grand',
        'name': 'Sarcoma',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_Liver': {
        'type': 'Grand',
        'name': 'Liver',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_TibialNerve': {
        'type': 'Grand',
        'name': 'Tibial_Nerve',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
                'original_mask': False,
                'num_node_features': 340,
                'num_classes': 3,
                'is_regression': False,
               }
    },
    'grand_KidneyCarcinoma': {
        'type': 'Grand',
        'name': 'Kidney_renal_papillary_cell_carcinoma',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
             'num_node_features': 340,
             'num_classes': 3,
             'is_regression': False,
               }
    },
    'grand_Spleen': {
        'type': 'Grand',
        'name': 'Spleen',
        'root': 'data/grand',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 340,
            'num_classes': 3,
            'is_regression': False,
               }
    },
    'musae_Twitch_DE':{
        'type': 'Twitch',
        'name': 'DE',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
    'musae_Twitch_EN':{
        'type': 'Twitch',
        'name': 'EN',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
     'musae_Twitch_ES':{
        'type': 'Twitch',
        'name': 'ES',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
        'musae_Twitch_FR':{
        'type': 'Twitch',
        'name': 'FR',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
    'musae_Twitch_PT':{
        'type': 'Twitch',
        'name': 'PT',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
     'musae_Twitch_RU':{
        'type': 'Twitch',
        'name': 'RU',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 2,
            'is_regression': False,
        }
    },
    'musae_Facebook':{
        'type': 'Facebook',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 4,
            'is_regression': False,
        }
    },
    'musae_Github':{
        'type': 'GitHub',
        'root': 'data/musae',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'num_classes': 4,
            'is_regression': False,
        }
    },
    'musae_Wiki_chameleon':{
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name':'chameleon',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
        }
    },
    'musae_Wiki_crocodile':{
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name':'crocodile',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
        }
    },
    'musae_Wiki_squirrel':{
        'type': 'Wikipedia',
        'root': 'data/musae',
        'name':'squirrel',
        'single_graph': True,
        'info':{
            'original_mask': False,
            'num_node_features': 128,
            'is_regression': True,
        }
    },

    'amazon':{
        'type': 'Amazon-place-holder',
        'name': 'place-holder'
    }
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
            shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler)
        self.single_graph = single_graph
        self.masks = masks
        self.onehot = onehot
        self.workers = workers
        #TODO sampler is potentially needed
        # self.sampler = None
        # self.batch_sampler = None

    @staticmethod
    def worker_init(x):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    #TODO implemetn hyper-graph partition
    def __iter__(self):
        if self.single_graph:
            for i, item in enumerate(super().__iter__()):
                yield item
        for i, item in enumerate(super().__iter__()):
            if self.onehot:
                item.y = torch.argmax(item.y, dim=1)
            yield (item.to(device), None)

def get_dataset(name, original_mask=False, split=0.9, batch_size=1, workers=2):
    # fix random seeds
    np.random.seed(1)
    torch.manual_seed(1)

    info = dict(DATASET_INFO[name])
    dataset_info = info.pop('info', {})

    single_graph = info.pop('single_graph', False)
    onehot = info.pop('onehot', False)
    if not single_graph:
        train_bsize = info.pop('train_batch_size', None)
        eval_bsize = info.pop('eval_batch_size', None)
        test_bsize = info.pop('test_batch_size', None)

    cls = getattr(datasets, info.pop('type'))
    print(info)
    dataset = cls(**info)
    kwargs = {
        'batch_size': batch_size,
        'workers': workers,
        'single_graph': single_graph,
    }

    if single_graph:
        original_mask = dataset_info.pop('original_mask')
        Loader = functools.partial(DataLoader, **kwargs)
        dataset, masks = mask_split(dataset, original_mask)
        # take one sample mask out
        train_mask, eval_mask, test_mask = masks[0]
        dataset = dataset[0]
        dataset.train_mask = train_mask
        dataset.val_mask = eval_mask
        dataset.test_mask = test_mask
        # dataloader requires a list of dataset
        dataset = [dataset]
        #logging.info(
        print(
            f"Search with a partition of {train_mask.sum()} train data, "
            f"{eval_mask.sum()} val data and {test_mask.sum()} test data.")
        # for single graph the masks is of no use
        print(dataset_info)
        return Loader(dataset, masks), Loader(dataset, masks), Loader(dataset, masks), dataset_info
    """
    train_set = dataset
    eval_set = cls(path, split='val', **info)
    test_set = cls(path, split='test', **info)
    train_num, eval_num, test_num = len(train_set), len(eval_set), len(test_set)

    kwargs['onehot'] = onehot
    kwargs['batch_size'] = train_bsize
    kwargs['shuffle'] = True
    TrainLoader = functools.partial(DataLoader, **kwargs)
    kwargs['batch_size'] = eval_bsize
    EvalLoader = functools.partial(DataLoader, **kwargs)
    kwargs['batch_size'] = test_bsize
    TestLoader = functools.partial(DataLoader, **kwargs)

    logging.info(
        f"Multi-graph search, "
        f"Search with a partition of {train_num} train graphs, "
        f"{eval_num} val graphs and {test_num} test graphs.")
    return (
        TrainLoader(train_set, None),
        EvalLoader(eval_set, None),
        TestLoader(test_set, None),
        info)
    """

def mask_split(
        dataset, original_mask=True,
        train_portion=0.6, eval_portion=0.2, test_portion=0.2):
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

def random_edge_split(data, train_ratio=0.6, val_ratio=0.2):
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
