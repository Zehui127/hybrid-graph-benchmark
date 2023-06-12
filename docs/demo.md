# HGB Training and Evaluation Frameworks

You can use the command `hybrid-graph` to train and evaluate GNNs predefined in HGB directly, if you have [installed HGB via pip](https://github.com/Zehui127/hybrid-graph-benchmark/#pip-install).

## Train

Training can be triggered with the following command. It takes only a few minutes to train a GCN even on a CPU.

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

### Command-Line Arguments

The command-line arguments are stored in the `arguments` dictionary as follows:

* `action`: Name of the action to perform. (train, eval)
* `dataset`: Name of the dataset. (`grand_Lung`, `musae_Facebook`, `amazon_Computer`, ...) Check [below](#available-datasets) for all the options.
* `model`: Name of the model. (`gcn`, `sage`, `gat`, ...) Check [below](#available-models) for all the options.

### Optional arguments:


* `-load, --load-name`: Name of the saved model to restore.
* `-save, --save-name`: Name of the saved model to save.
* `-opt, --optimizer`: Pick an optimizer.
* `-lr, --learning-rate`: Initial learning rate.
* `-m, --max-epochs`: Maximum number of epochs for training.
* `-b, --batch-size`: Batch size for training and evaluation.
* `-d, --debug`: Verbose debug.
* `-seed, --seed`: Number of steps for model optimization.
* `-w, --num_workers`: Number of CPU workers.
* `-n, --num_devices`: Number of GPU devices.
* `-a, --accelerator`: Accelerator style. (cpu, gpu, tpu)
* `-s, --strategy`: Strategy style. (ddp, ddp2, ddp_spawn, ddp_cpu, ddp_sharded, dp, horovod, single)

## Available Datasets

1. `grand_ArteryAorta`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains a graph of the human aorta artery.
2. `grand_ArteryCoronary`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains a graph of the coronary arteries.
3. `grand_Breast`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains information about breast tissue.
4. `grand_Brain`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains information about brain tissue.
5. `grand_Lung`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains information about lung tissue.
6. `grand_Stomach`: belong to one of [GRAND-Tissues](datasets.md#grand-tissues) - This dataset contains information about stomach tissue.
7. `grand_Leukemia`: belong to one of [GRAND-Tissues](datasets.md#grand-diseases) - This dataset contains information about leukemia.
8. `grand_Lungcancer`: belong to one of [GRAND-Tissues](datasets.md#grand-diseases) - This dataset contains information about lung cancer.
9. `grand_Stomachcancer`: belong to one of [GRAND-Tissues](datasets.md#grand-diseases) - This dataset contains information about stomach cancer.
10. `grand_KidneyCancer`: belong to one of [GRAND-Tissues](datasets.md#grand-diseases) - This dataset contains information about kidney cancer.
11. `musae_Twitch_DE`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in German.
12. `musae_Twitch_EN`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in English.
13. `musae_Twitch_ES`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in Spanish.
14. `musae_Twitch_FR`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in French.
15. `musae_Twitch_PT`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in Portuguese.
16. `musae_Twitch_RU`: belong to one of [MUSAE-Twitch](datasets.md#musae-twitch) - This dataset contains a graph of Twitch streamers in Russian.
17. `musae_Facebook`: belong to one of [MUSAE-Facebook](datasets.md#musae-facebook) - This dataset contains a graph of Facebook users.
18. `musae_Github`: belong to one of [MUSAE-GitHub](datasets.md#musae-github) - This dataset contains a graph of GitHub users.
19. `musae_Wiki_chameleon`: belong to one of [MUSAE-Wiki](datasets.md#musae-wiki) - This dataset contains a graph of Wikipedia users editing pages about chameleons.
20. `musae_Wiki_crocodile`: belong to one of [MUSAE-Wiki](datasets.md#musae-wiki) - This dataset contains a graph of Wikipedia users editing pages about crocodiles.
21. `musae_Wiki_squirrel`: belong to one of [MUSAE-Wiki](datasets.md#musae-wiki) - This dataset contains a graph of Wikipedia users editing pages about squirrels.
22. `amazon_Computer`: belong to one of [Amazon-Computers](datasets.md#amazon-datasets) - This dataset contains a graph of Amazon computer products.
23. `amazon_Photo`: belong to one of [Amazon-Photos](datasets.md#amazon-datasets) - This dataset contains a graph of Amazon products related to photography.

## Available Models

1. `gcn`: **Graph Convolutional Network (GCN)** - A classical type of GNN that applies the convolution operation to graphs.

2. `sage`: **GraphSAGE (Sample and Aggregate)** - Another type of GNN which learns an embedding by aggregating information from a node's local neighborhood.

3. `gat`: **Graph Attention Network (GAT)** - A classsical type of GNN which applies attention mechanisms to weigh the importance of neighboring nodes when aggregating their information.

4. `gatv2`: **GATv2** - An improved version of GAT.

5. `hyper-gcn`: **Hypergraph Convolution (HyperConv)** - A type of GNN designed to work with hypergraphs, where an edge can connect more than two nodes.

6. `hyper-gat`: **Hypergraph Attention (HyperAtten)** - An attention-based model similar to GAT, but designed for hypergraphs.

<!-- 7. `ensemble`: **Average Ensemble** - This model combines the predictions of multiple other models by averaging their outputs. -->

7. `lp-gcn-hyper-gcn`: **LP-GCN+HyperConv** - A Linear Probe model that combines GCN and HyperConv.

8. `lp-gat-hyper-gcn`: **LP-GAT+HyperConv** - A Linear Probe model that combines GAT and HyperGCN.

9. `lp-gat-gcn`: **LP-GAT+GCN** - A Linear Probe model that combines GAT and GCN.

<!-- 11. `lp-gcn-gcn`: **LP-GCN+GCN** - A Linear Probe model that combines two GCNs.

12. `lp-gat-gat`: **LP-GAT+GAT** - A Linear Probe model that combines two GATs.

13. `lp-hyper-hyper`: **LP-Hyper+Hyper** - A Linear Probe model that combines two HyperGCN or HyperGAT models. -->
