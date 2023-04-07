# Hypergraph Benchmarks
[![paper](https://img.shields.io/badge/Paper-Open%20Review-orange)]()
&nbsp;&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/Access-PyTorch%20Geometric-green)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

![](https://github.com/Zehui127/zehui127/blob/main/images/icon2.png?raw=true)

This is a benchmark dataset for the hypergraph learning, including various hypergraphs with ported hypergraph learning algorithms.

# Data Specification

we use the ```torch_geometric.data.Data``` to wrap the graphs with additional adjacency matrix for hyperedge representation.

## Hyperedge representation
The format of hyper edge follows the following [convention:](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv) each node $v$ is mapped to a hyperedge $e$ $[v,e]$
<details open>
<summary><b>For example</b></summary>

>   In the hypergraph scenario
>     A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with
>     vertices $\mathcal{V} = \\{ 0, 1, 2, 3 \\}$ and
>     two hyperedges $\mathcal{E} = \\{ \\{ 0, 1, 2 \\}, \\{ 1, 2, 3 \\} \\}$,
>     the `hyperedge_index` is represented as:

>Example:
>```
>hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3],
>                                 [0, 0, 0, 1, 1, 1]])
>```
</details>

## Data
The graphs are wrapped with ```torch_geometric.data.Data```.

<details open>
<summary><b> Specification </b></summary>

> ```x:``` the node embedding  
> ```y:``` the classes of y  
> ```adj:``` the edge_index of graph  
> ```sparse_adj:``` the sparse format of edge_index  
> ```hyperedge_index:``` the hyperedge index of the graph  
> ```sparse_hyperedge_index:``` the sparse format of hyperedge index  

>Example:
>```
># grand gene regulatory network
>Graph = Data(x=[30171, 340], y=[30171], adj=[2, 2496126],
>              sparse_adj=[30171, 30167, nnz=2496126],
>              hyperedge_index=[2, 72098],
>              sparse_hyperedge_index=[30170, 38170, nnz=72098])
>```
</details>



# Benchmarks
## Gene regulatory networks
There are 36 gene regulatory networks for tissues and 24 networks for diseases. The hypergraphs are constructed by connecting nearby genes in the chromosone with hyper edges.

## Social Networks
There are 8 social networks derived from the Facebook pages, GitHub developers, and Twitch gamers in the ["Multi-scale Attributed Node Embedding" (MUSAE)](https://arxiv.org/abs/1909.13021) paper. The hypergraphs are mutually connected sub-groups that contain at least 3 nodes (i.e., maximal cliques with sizes of at least 3). Each dataset has an option to use either the raw node features, or the preprocessed node embeddings as introduced in the MUSAE paper.

## Wikipedia Networks
There are 3 English Wikipedia page-page networks on specific topics (chameleons, crocodiles and squirrels) collected in December 2018, based on the ["Multi-scale Attributed Node Embedding" (MUSAE)](https://arxiv.org/abs/1909.13021) paper. The hypergraphs are mutually connected page groups that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3). Each dataset has an option to use either the raw node features, or the preprocessed node embeddings as introduced in the MUSAE paper.

## Amazon Reviews

# Ported Learning Algorithms
1. Hypergraph Convolution and Hypergraph Attention (To be ported)
    * Paper: [link](https://arxiv.org/abs/1901.08150)
    * Implementation: [link](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv)

# Training
The training workflow is written in pytorch lightning, more instruction to add later.
Currrent traing only work with the following argument:
```
python hg.py train grand1 toynet
```
## PyG

- Installing PyG

    ```bash
    TORCH=1.13.0
    CUDA=cpu #cu111... depending on the cuda version on your system
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    python -m pip install torch-geometric
    ```

- training
    ```
     python hg.py train grand1 toynet # training
    ```

- evaluation
    ```bash
     python hg.py eval cora toynet --load ./lightning_logs/...
    ```
- check available arugments with the following
    ```bash
     python hg.py 
    ```

# FAQs
Q: I got ```_pickle.UnpicklingError: Failed to interpret file '*.npz' as a pickle``` when I try to load the ```musae``` datasets. How to solve it?  
A: Simply delete the downloaded files and try loading again.
