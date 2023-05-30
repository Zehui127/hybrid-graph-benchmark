<p align='center'>
<img src="https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/docs/title.png?raw=true" style="width: 75%; height: auto;"/>
</p>

------------------------------------------------------------------

[![paper](https://img.shields.io/badge/Download-Raw%20Data-green)](https://zenodo.org/record/7982540)
[![PyPI version](https://img.shields.io/pypi/v/hybrid-graph?color=purple)](https://pypi.org/project/hybrid-graph/)
[![license](https://img.shields.io/github/license/Zehui127/hybrid-graph-benchmark)](LICENSE)

<!-- [![paper](https://img.shields.io/badge/Paper-Open%20Review-orange)]() -->
<!-- [![paper](https://img.shields.io/badge/Access-PyTorch%20Geometric-green)](https://pytorch-geometric.readthedocs.io/en/latest/index.html) -->
<!-- ![]() -->

This is a benchmark dataset for evaluating **hybrid-graph** (hypergraph and hierarchical graph) learning algorithms. It contains:
 - 23 real-world higer-order graphs from the domains of biology, social media, and wikipedia
 - Built-in functionalities for preprocessing hybrid-graphs
 - A framework to easily train and evaluate Graph Neural Networks
<!-- ![](https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/img/architecture.png?raw=true) -->
<img src="https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/docs/architecture.png?raw=true" style="width: 75%; height: auto;">


# Installation
## Requirements
First, install the required PyTorch packages. You will need to know the version of CUDA you have installed, as well as the version of PyTorch you want to use. Replace `${TORCH}` and `${CUDA}` with these versions in the following commands:

```bash
# TORCH=2.0.1 if use newest stable torch version
# CUDA=cpu if cuda is not available
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric==2.2.0
```
Once these dependencies are installed, you can install this package with one of the following:
## Pip install
```bash
pip install hybrid-graph
# or pip install git+https://github.com/Zehui127/hybrid-graph-benchmark.git
```
## From source
```bash
git clone https://github.com/Zehui127/hybrid-graph-benchmark.git
cd hybrid-graph-benchmark
pip install -e .
```

# Usage
Hybrid-graph provide both datasets and flash training/evaluation capabilities.
## (1) Access the Dataset
we use the ```torch_geometric.data.Data``` to wrap the graphs with additional adjacency matrix for hyperedge representation.
```python
from hg.datasets import Facebook, HypergraphSAINTNodeSampler
# download data to the path 'data/facebook'
data = Facebook('data/facebook')
print(data[0]) # e.g. Data(x=[22470, 128], edge_index=[2, 342004], y=[22470], hyperedge_index=[2, 2344151], num_hyperedges=236663)

# create a sampler which sample 1000 nodes from the graph for 5 times
sampler = HypergraphSAINTNodeSampler(data[0],batch_size=1000,num_steps=5)
batch = next(iter(sampler))
print(batch)  # e.g. Data(num_nodes=918, edge_index=[2, 7964], hyperedge_index=[2, 957528], num_hyperedges=210718, x=[918, 128], y=[918])
```
Data Loaders can also be obtained using ```hg.hybrid_graph.io.get_dataset```
```python
from hg.hybrid_graph.io import get_dataset
name = 'musae_Facebook'
train_loader, valid_loader, test_loader,data_info = get_dataset(name)
```
## (2) Train/Evaluate with ```hybrid-graph```
Assuming that you have [Pip install](#pip-install).

Training can be triggered with the following, it takes only a few minutes to train GCN on even on CPU device.
```bash
#-a=gpu,cpu,tpu
hybrid-graph train grand_Lung gcn -a=cpu
```
Evaluation can be triggered with
```bash
# load the saved checkpoint from the path 'lightning_logs/version_0/checkpoints/best.ckpt'
hybrid-graph eval grand_lung gcn -load='lightning_logs/version_0/checkpoints/best.ckpt' -a=cpu
```

# Add New Models
In order to add new models, you should [Install from source](#from-source).
```bash
cd hybrid-graph-benchmark/hg/hybrid_graph/models/gnn
touch customize_model.py
```
Within ```customize_model.py```, it should correctly handle the input feature size, prediction size and task type.
Below is the definition of vanila Graph Convolutional Networks (GCN)
```python
from torch_geometric.nn import GCNConv
class CustomizeGNN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.conv2 = GCNConv(dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
```
Finally, you should register you model in ```hybrid-graph-benchmark/hg/hybrid_graph/models/__init__.py```

```python
from .gnn.customize_model import CustomizeGNN
factory = {
            'sage': SAGENet,
            'gcn':CustomizeGNN, # abbreviation: ClassName,
          }
```
