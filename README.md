<p align='center'>
<img src="https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/img/title.png?raw=true" style="width: 80%; height: auto;"/>
</p>

------------------------------------------------------------------

[![paper](https://img.shields.io/badge/Download-Raw%20Data-green)]()
<!-- [![paper](https://img.shields.io/badge/Paper-Open%20Review-orange)]() -->
<!-- [![paper](https://img.shields.io/badge/Access-PyTorch%20Geometric-green)](https://pytorch-geometric.readthedocs.io/en/latest/index.html) -->

<!-- ![]() -->

This is a benchmark dataset for evaluating **hybrid-graph** (hypergraph and hierarchical graph) learning algorithms. It contains:
 - 23 real-world higer-order graphs from the domains of biology, social media, and wikipedia
 - Built-in functionalities for preprocessing hybrid-graphs
 - A framework to easily train and evaluate Graph Neural Networks
<!-- ![](https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/img/architecture.png?raw=true) -->
<img src="https://github.com/Zehui127/hypergraph-benchmarks/blob/pre-release/img/architecture.png?raw=true" style="width: 90%; height: auto;">


# Installation
## Requirements
First, install the required PyTorch packages. You will need to know the version of CUDA you have installed, as well as the version of PyTorch you want to use. Replace `${TORCH}` and `${CUDA}` with these versions in the following commands:

```bash
# TORCH=1.13
# CUDA=cpu if cuda is not available
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric==2.2.0
```
Once these dependencies are installed, you can install this package with:
## Pip install
```bash
pip install hybrid-graph
```
## From source
```bash
git clone https://github.com/Zehui127/hypergraph-benchmarks.git
cd hypergraph-benchmarks
pip install -e .
```

# Usage

# Add New Datasets or New GNNs
