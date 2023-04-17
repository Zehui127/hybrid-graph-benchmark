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
>from datasets import grand
>datasets = grand.Grand("data/grand",'Artery_Aorta')
>datasets[0]
>Data(x=[30171, 340], y=[30171], edge_index=[2, 2080169],
>              hyperedge_index=[2, 72098],num_hyperedges=29956)
>```
</details>



# Benchmarks
## Gene regulatory networks
10 gene regulatory networks for tissues and diseases are selected and preprocessed from [GRAND: a database of gene regulatory network models across human conditions](https://grand.networkmedicine.org). The hypergraphs are constructed by connecting nearby genes in the chromosone with hyper edges. Each node represets a gene, the node embedding of 340 is created using [k-mer analysis](https://en.wikipedia.org/wiki/K-mer). In this analysis, all possible k-mers with k values ranging from 1 to 4 are enumerated, and the counts of these k-mers serve as the elements of the feature vector.
## Social Networks
There are 8 social networks derived from the Facebook pages, GitHub developers, and Twitch gamers in the ["Multi-scale Attributed Node Embedding" (MUSAE)](https://arxiv.org/abs/1909.13021) paper. The hypergraphs are mutually connected sub-groups that contain at least 3 nodes (i.e., maximal cliques with sizes of at least 3). Each dataset has an option to use either the raw node features, or the preprocessed node embeddings as introduced in the MUSAE paper.

## Wikipedia Networks
There are 3 English Wikipedia page-page networks on specific topics (chameleons, crocodiles and squirrels) collected in December 2018, based on the ["Multi-scale Attributed Node Embedding" (MUSAE)](https://arxiv.org/abs/1909.13021) paper. The hypergraphs are mutually connected page groups that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3). Each dataset has an option to use either the raw node features, or the preprocessed node embeddings as introduced in the MUSAE paper.

## Amazon Reviews

# Evaluated Algorithms
## Node Classification

| Model     | Description                                                                                                                                                                                                                                                        | Accuracy on Grand | Accuracy on Social Media Network | Accuracy on Amazon Reviews |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|----------------------------------|----------------------------|
| GCNs      | [Graph Convulutional Networks](https://arxiv.org/abs/1609.02907) on original graphs without using hyperedges/hierarchical information.                                                                                                                             |                   |                                  |                            |
| GAT       | [Graph Attention Networks](https://personal.utdallas.edu/~fxc190007/courses/20S-7301/GAT-questions.pdf) on original graphs without using hyperedges/hierarchical information.                                                                                      |                   |                                  |                            |
| GraphSage | [GraphSAGE](https://arxiv.org/abs/1706.02216) on original graphs without using hyperedges/hierarchical information.                                                                                                                                                |                   |                                  |                            |
| HyperGCNs | [Hypergraph Convolution](https://arxiv.org/abs/1901.08150)[ (link)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv) only uses hyper edges to aggregate information |                   |                                  |                            |
| HyperGAT  | [Hypergraph Attention](https://arxiv.org/abs/1901.08150)[ (link)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv) uses hyper edges to aggregate information        |                   |                                  |                            |


# Training
The training workflow is written in pytorch lightning, the new models can be added in the ```hybrid_graph/models/gnn``` and register you model in ```hybrid_graph/models/gnn/__init__.py```.

Currrent traing works for all datasets with gcn and sage mdoels:
```
python hg.py train grand_ArteryAorta gcn
python hg.py train grand_Breast sage
python hg.py train musae_Twitch_ES gcn
python hg.py train musae_Twitch_FR sage
python hg.py train musae_Facebook gcn
python hg.py train musae_Facebook sage
python hg.py train musae_Github gcn
python hg.py train musae_Github sage
```

for eval
```
# ./lightning_logs/... should the path where the checkpoints are saved
python hg.py eval grand_ArteryAorta gcn --load ./lightning_logs/...
python hg.py eval grand_ArteryAorta sage --load ./lightning_logs/...
```

# FAQs
Q: I got ```_pickle.UnpicklingError: Failed to interpret file '*.npz' as a pickle``` when I try to load the ```musae``` datasets. How to solve it?
A: Simply delete the downloaded files and try loading again.
