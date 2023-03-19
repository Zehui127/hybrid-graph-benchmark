# hypergraph-benchmarks
A benchmark of a collections of datasets for hypergraph tasks

# Grand - gene regulatory networks
There are 36 gene regulatory networks for tissues and 27 networks for diseases. The hypergraphs are constructed by connecting nearby genes in the chromosone with hyper edges.

#### Requirement
```
pip3 -r install grand/requirements.txt
```

#### Usage
Change the ```path``` property in grand_config.json to the directory where you want to store grand graphs

Run ```GrandGraph()``` to download the graphs
```
from grand import GrandGraph
network = GrandGraph("Adipose_Subcutaneous","https://granddb.s3.amazonaws.com/tissues/networks/Adipose_Subcutaneous.csv")
```

And run ```get_dataset_ensembl_ids()``` to get hyper_edge
```
from grand import get_dataset_ensembl_ids
hyper_edge = get_dataset_ensembl_ids()
```

Finally run ```GenerateHyperGraph() ``` to generate formatted hypergraphs; it will return new_original graph and the hypergraph
```
TODO GenerateHyperGraph(network,hyper_edge) is not implemented yet
```
#### Data Specification
##### Graph Structure
Mixed representation of graphs
```
Graph = Data(x=x_emd, y=class, [adj,hyper_edge])
```
where ```x_emd``` will be generated through enformer; node class will be the following:
##### Node Class
In the original graph, the node representation will be generated using Enformer. There are two set of defined tasks:
1. Edge prediction: Mask part of the edges and predict the rest of the edges
2. Node classfication: classify if a node (gene) is protein-coding or not given 36+24 graphs.
# Useful links

A collection of [datasets benchmarks papers](https://nips.cc/virtual/2022/events/datasets-benchmarks-2022) from NeurIPS 2022
