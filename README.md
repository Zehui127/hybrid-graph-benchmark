# hypergraph-benchmarks
A benchmark of a collections of datasets for hypergraphs
![hypergraph](https://ibb.co/f2VCvdP)

[![paper](https://img.shields.io/badge/Paper-Open%20Review-orange)]()
&nbsp;&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/Access-PyTorch%20Geometric-green)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)


# Data Specification

we use the ```torch_geometric.data.Data``` to wrap the graphs with additional adjacency matrix for hyperedge representation.

## Hyperedge representation
The format of hyper edge follows the following [convention:](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv) each node $v$ is mapped to a hyperedge $e$: $[v,e]$
<details open>
<summary><b>For example</b></summary>

>   In the hypergraph scenario
>     A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with
>     vertices $\mathcal{V} = \{ 0, 1, 2, 3 \}$ and
>     two hyperedges $\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}$,
>     the `hyperedge_index` is represented as:

>```
>hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3],
>                                 [0, 0, 0, 1, 1, 1]])
>```
</details>
## Data

The graphs are wrapped with ```torch_geometric.data.Data```. Each graph includes the following properties:
```x:``` the node embedding
```y:``` the classes of y
```adj:``` the edge_index of graph
```sparse_adj:``` the sparse format of edge_index
```hyperedge_index:``` the hyperedge index of the graph
```sparse_hyperedge_index:``` the sparse format of hyperedge index
An example is included as below:
```
Graph = Data(x=[30171, 340], y=[30171], adj=[2, 2496126],
             sparse_adj=[30171, 30167, nnz=2496126],
             hyperedge_index=[2, 72098],
             sparse_hyperedge_index=[30170, 38170, nnz=72098])
# convert the edge_index to sparse format
sparse_adj = SparseTensor.from_edge_index(adj,weights)
```

# Datasets
## Gene regulatory networks
There are 36 gene regulatory networks for tissues and 24 networks for diseases. The hypergraphs are constructed by connecting nearby genes in the chromosone with hyper edges.

## Social Networks

## Amazon Reviews

# Useful links

A collection of [datasets benchmarks papers](https://nips.cc/virtual/2022/events/datasets-benchmarks-2022) from NeurIPS 2022
