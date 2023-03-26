# hypergraph-benchmarks
[![paper](https://img.shields.io/badge/Paper-Open%20Review-orange)]()
&nbsp;&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/Access-PyTorch%20Geometric-green)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

![](https://github.com/Zehui127/zehui127/blob/main/images/icon2.png?raw=true)

This is a benchmark dataset for the hypergraph learning, including various hypergraphs with ported hypergraph learning algorithms.

# How to get started

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
>             sparse_adj=[30171, 30167, nnz=2496126],
>             hyperedge_index=[2, 72098],
>             sparse_hyperedge_index=[30170, 38170, nnz=72098])
>```
</details>



# Benchmarks
## Gene regulatory networks
There are 36 gene regulatory networks for tissues and 24 networks for diseases. The hypergraphs are constructed by connecting nearby genes in the chromosone with hyper edges.

## Social Networks

## Amazon Reviews

# Ported Learning Algorithms
1. Hypergraph Convolution and Hypergraph Attention (To be ported)
    * Paper: [link](https://arxiv.org/abs/1901.08150)
    * Implementation: [link](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv)
