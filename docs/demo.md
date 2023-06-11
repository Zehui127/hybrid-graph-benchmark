# Modules for Training and Evaluation
Assuming that you have [Pip install](https://github.com/Zehui127/hybrid-graph-benchmark/#pip-install).
## Train
Training can be triggered with the following, it takes only a few minutes to train GCN even on CPU device.
```bash
#-a=gpu,cpu,tpu
hybrid-graph train grand_Lung gcn -a=cpu
```
## Evaluate
Evaluation can be triggered with
```bash
# load the saved checkpoint from the path 'lightning_logs/version_0/checkpoints/best.ckpt'
hybrid-graph eval grand_lung gcn -load='lightning_logs/version_0/checkpoints/best.ckpt' -a=cpu
```

## Command Line Interface
===============

This class is responsible for parsing the command-line arguments and setting up the training and testing environment.

Command-line Arguments
----------------------

The command-line arguments are stored in the `arguments` dictionary as follows:

- `action`: Name of the action to perform.
- `dataset`: Name of the dataset.
- `model`: Name of the model.

Optional arguments:
-------------------

- `-load, --load-name`: Name of the saved model to restore.
- `-save, --save-name`: Name of the saved model to save.
- `-opt, --optimizer`: Pick an optimizer.
- `-lr, --learning-rate`: Initial learning rate.
- `-m, --max-epochs`: Maximum number of epochs for training.
- `-b, --batch-size`: Batch size for training and evaluation.
- `-d, --debug`: Verbose debug.
- `-seed, --seed`: Number of steps for model optimization.
- `-w, --num_workers`: Number of CPU workers.
- `-n, --num_devices`: Number of GPU devices.
- `-a, --accelerator`: Accelerator style. (cpu, gpu, tpu)
- `-s, --strategy`: Strategy style. (ddp, ddp2, ddp_spawn, ddp_cpu, ddp_sharded, dp, horovod, single)
