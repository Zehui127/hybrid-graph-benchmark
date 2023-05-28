import sys
import random
import argparse
import functools
import logging

import os
import torch
import numpy as np

from .sessions import train, test
from .models import factory
from .io import get_dataset
import pathlib


class Main:
    arguments = {
        ('action', ): {'type': str, 'help': 'Name of the action to perform.'},
        ('dataset', ): {'type': str, 'help': 'Name of the dataset.'},
        ('model', ): {'type': str, 'help': 'Name of the model.'},

        # dataset path
        ('-dp', '--dataset-path'): {
            'type': str, 'default': os.path.join(pathlib.Path(__file__).parent.parent.resolve(),'datasets'), 'help': 'Path to the dataset.',
        },

        # checkpointing
        ('-load', '--load-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to save.'
        },

        # common training args
        ('-opt', '--optimizer'): {
            'type': str, 'default': 'adam', 'help': 'Pick an optimizer.',
        },
        ('-lr', '--learning-rate'): {
            'type': float, 'default': 0.01, 'help': 'Initial learning rate.',
        },
        ('-m', '--max-epochs'): {
            'type': int, 'default': 100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch-size'): {
            'type': int, 'default': 1,
            'help': 'Batch size for training and evaluation.',
        },

        # debug control
        ('-d', '--debug'): {
            'action': 'store_true', 'help': 'Verbose debug',
        },
        ('-seed', '--seed'): {
            'type': int, 'default': 2, 'help': 'Number of steps for model optimisation',
        },

        # cpu gpu setup
        ('-w', '--num_workers'): {
            # multiprocessing fail with too many works
            'type': int, 'default': 0, 'help': 'Number of CPU workers.',
        },
        ('-n', '--num_devices'): {
            'type': int, 'default': 1, 'help': 'Number of GPU devices.',
        },
        ('-a', '--accelerator'): {
            'type': str, 'default': 'gpu', 'help': 'Accelerator style.',
        },
        ('-s', '--strategy'): {
            'type': str, 'default': 'ddp', 'help': 'strategy style.',
        }
    }
    def __init__(self):
        super().__init__()
        a = self.parse()
        if a.debug:
            sys.excepthook = self._excepthook
        # seeding
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        self.a = a

    def parse(self):
        p = argparse.ArgumentParser(description='Genome Graph.')
        for k, v in self.arguments.items():
            p.add_argument(*k, **v)
        return p.parse_args()

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        import ipdb
        ipdb.post_mortem(etb)

    def run(self):
        try:
            action = getattr(self, f'cli_{self.a.action.replace("-", "_")}')
        except AttributeError:
            callables = [n[4:] for n in dir(self) if n.startswith('cli_')]
            logging.error(
                f'Unkown action {self.a.action!r}, '
                f'accepts: {", ".join(callables)}.')
        return action()

    def setup_model_and_data(self, a):
        # get dataset
        train_loader, val_loader, test_loader, dataset_info = get_dataset(
            name=a.dataset, batch_size=a.batch_size, workers=a.num_workers,
            datasets_path=a.dataset_path)
        print(f"datasets path: {a.dataset_path}")
        # get model
        model_cls = factory[a.model]
        model = model_cls(info=dataset_info)
        return model, train_loader, val_loader, test_loader, dataset_info

    def cli_train(self):
        a = self.a
        if not a.save_name:
            logging.error('--save-name not specified.')

        model, train_loader, val_loader, test_loader, dataset_info = self.setup_model_and_data(a)

        plt_trainer_args = {
            'max_epochs': a.max_epochs, 'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,
            'fast_dev_run': a.debug,}

        train_params = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': a.optimizer,
            'learning_rate': a.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "save_path": a.save_name,
            "dataset_info": dataset_info,
            "seed": a.seed,
        }
        train(**train_params)

    def cli_test(self):
        a = self.a

        model, train_loader, val_loader, test_loader, dataset_info = self.setup_model_and_data(a)
        plt_trainer_args = {
            'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,}
        test_params = {
            'model': model,
            'test_loader': test_loader,
            'plt_trainer_args': plt_trainer_args,
            'load_path': a.load_name,
            'dataset_info': dataset_info,
        }
        test(**test_params)
    cli_eval = cli_test

# hybrid_graph/cli.py
def main():
    Main().run()
