import os
import torch
import pytorch_lightning as pl

from .plt_wrapper import ModelWrapper


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def test(model, test_loader, plt_trainer_args, load_path, dataset_info):
    plt_model = ModelWrapper(model, dataset_info)
    print(load_path)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    #randomize_weights(plt_model)
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_loader)

def randomize_weights(module, a=-1, b=1):
    """
    Randomize weights of a given PyTorch module using uniform distribution.

    Args:
        module (nn.Module): The PyTorch module whose weights will be randomized.
        a (float, optional): Lower bound of the uniform distribution. Default is -1.
        b (float, optional): Upper bound of the uniform distribution. Default is 1.
    """
    for name, param in module.named_parameters():
        if param.requires_grad and 'weight' in name:
            print(name)
            torch.nn.init.uniform_(param.data, a=a, b=b)
        elif param.requires_grad and 'bias' in name:
            print(name)
            torch.nn.init.zeros_(param.data)
