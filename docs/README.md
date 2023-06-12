
<p align='center'>
<img src="title.png?raw=true" style="width: 100%; height: auto;"/>
</p>


This is a benchmark dataset for evaluating **hybrid graph** (a unified definition for higher-order graphs, including hypergraphs and hierarchical graphs) learning algorithms. It contains:

* 23 real-world higer-order graphs from the domains of social media (MUSAE), biology (GRAND), and e-commerce (Amazon).
* Built-in functionalities for preprocessing hybrid-graphs
* A framework to easily train and evaluate Graph Neural Networks

<p align='left'>
<img src="architecture.png?raw=true" style="width: 70%; height: auto;"/>
</p>

### Get started by installing from [GitHub](https://github.com/Zehui127/hybrid-graph-benchmark/)

# Modules

### [Hybrid Graph Datasets](datasets.md#musae-github)
### [Hybrid Graph Evaluation Framework](demo.md#train)

# Summary 

Click the dataset name to see more details:

| Name                                             | #Graphs | #Nodes | #Edges | #Hyperedges | Avg. Node Degree | Avg. Hyperedge Degree | Clustering Coef. | Task Type           |
|--------------------------------------------------|--------:|-------:|-------:|------------:|-----------------:|----------------------:|-----------------:|--------------------:|
| [MUSAE-GitHub](datasets.md#musae-github)         | 1       | 37,700 | 578,006| 223,672     | 30.7             | 4.6                   | 0.168            | Node Classification |
| [MUSAE-Facebook](datasets.md#musae-facebook)     | 1       | 22,470 | 342,004| 236,663     | 30.4             | 9.9                   | 0.360            | Node Classification |
| [MUSAE-Twitch](datasets.md#musae-twitch)         | 6       | 5,686  | 143,038| 110,142     | 50.6             | 6.0                   | 0.210            | Node Classification |
| [MUSAE-Wiki](datasets.md#musae-wiki)             | 3       | 6,370  | 266,998| 118,920     | 88.8             | 14.4                  | 0.413            | Node Regression     |
| [GRAND-Tissues](datasets.md#grand-tissues)       | 6       | 5,931  | 5,926  | 11,472      | 2.0              | 1.3                   | 0.000            | Node Classification |
| [GRAND-Diseases](datasets.md#grand-diseases)     | 4       | 4,596  | 6,252  | 7,743       | 2.7              | 1.3                   | 0.000            | Node Classification |
| [Amazon-Computers](datasets.md#computers--photos)| 1       | 10,226 | 55,324 | 10,226      | 10.8             | 4.0                   | 0.249            | Node Classification |
| [Amazon-Photos](datasets.md#computers--photos)   | 1       | 6,777  | 45,306 | 6,777       | 13.4             | 4.8                   | 0.290            | Node Classification |

# License

Source code: [MIT license](https://opensource.org/license/mit/)  
MUSAE & GRAND datasets: [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.html)  
Amazon datasets: [Amazon Service license](https://s3.amazonaws.com/amazon-reviews-pds/LICENSE.txt)  


# Cite This Project

```bibtex
@article{Li2023HybridGraph,
    title={Hybrid Graph: A Unified Graph Representation with Datasets and Benchmarks for Complex Graphs},
    author={Zehui Li and 
            Xiangyu Zhao and 
            Mingzhu Shen and
            Guy-Bart Stan and
            Pietro Li{\`o} and
            Yiren Zhao},
    journal={arXiv preprint arXiv:2306.05108},
    year={2023}
}
```
