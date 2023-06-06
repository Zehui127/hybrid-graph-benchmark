import os
import sys
import math
import signal
import functools
import numpy as np
import pathlib
import torch
import torch_geometric
import torch_geometric.utils as utils
import logging

from .utils import device
import yaml
import hg.datasets as datasets
torch.multiprocessing.set_sharing_strategy('file_system')


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


def get_dataset(name, datasets_path=os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(),'datasets'),
                original_mask=False, split=0.6, batch_size=6000, workers=2, num_steps=5):
    # if datasets_path not in sys.path:
    #     sys.path.append(datasets_path)
    # import datasets

    with open(os.path.join(datasets_path, 'dataset_info.yaml')) as f:
        DATASET_INFO = yaml.safe_load(f)

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
        dataset = datasets.create_edge_label(dataset)
    dataset, masks = datasets.mask_split(dataset, original_mask)
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
