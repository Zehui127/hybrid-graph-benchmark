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
### Command-line Arguments

The command-line arguments are stored in the `arguments` dictionary as follows:

- `action`: Name of the action to perform. (train, eval)
- `dataset`: Name of the dataset. (grand_Lung, musae_Facebook, amazon_Computer, ...) Check [below](#Available-Datasets) for all the options.
- `model`: Name of the model. (gcn, sage, gat,...) Check [below](#Available-Models) for all the options.

### Optional arguments:


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

## Available Datasets

[Details](datasets.md#musae-github)

1. grand_ArteryAorta
2. grand_ArteryCoronary
3. grand_Breast
4. grand_Brain
5. grand_Lung
6. grand_Stomach
7. grand_Leukemia
8. grand_Lungcancer
9. grand_Stomachcancer
10. grand_KidneyCancer
11. musae_Twitch_DE
12. musae_Twitch_EN
13. musae_Twitch_ES
14. musae_Twitch_FR
15. musae_Twitch_PT
16. musae_Twitch_RU
17. musae_Facebook
18. musae_Github
19. musae_Wiki_chameleon
20. musae_Wiki_crocodile
21. musae_Wiki_squirrel
22. amazon_Photo
23. amazon_Computer

## Available Models


1. `gcn`: **GCNNet** - Graph Convolutional Network (GCN) is a type of GNN that applies a form of convolution to graph structured data.

2. `sage`: **SAGENet** - GraphSAGE (Graph Sample and Aggregate) is another type of GNN which learns an embedding by aggregating information from a node's local neighborhood.

3. `gat`: **GATNet** - Graph Attention Network (GAT) is a type of GNN which applies attention mechanisms to weigh the importance of neighboring nodes when aggregating their information.

4. `gatv2`: **GATV2Net** - This is an improved version of Graph Attention Network (GAT).

5. `hyper-gcn`: **HyperGCN** - Hypergraph Convolutional Networks are a type of GNN designed to work with hypergraphs, where an edge can connect more than two nodes.

6. `hyper-gat`: **HyperGAT** - This could be an attention-based model similar to GAT but designed for hypergraphs.

7. `ensemble`: **Average_Ensemble** - This model likely combines the predictions of multiple other models by averaging their outputs.

8. `lp-gcn-hyper-gcn`: **LPGCNHyperGCN** - This model appears to be a Linear Probe model that combines GCN and HyperGCN.

9. `lp-gat-hyper-gcn`: **LPGATHyperGCN** - This is a Linear Probe model that combines GAT and HyperGCN.

10. `lp-gat-gcn`: **LPGGATGCN** - A Linear Probe model that combines GAT and GCN.

11. `lp-gcn-gcn`: **LPGCNGCN** - A Linear Probe model that combines two GCNs.

12. `lp-gat-gat`: **LPGGATGAT** - A Linear Probe model that combines two GATs.

13. `lp-hyper-hyper`: **LPHYPERHYPER** - A Linear Probe model that combines two HyperGCN or HyperGAT models.
