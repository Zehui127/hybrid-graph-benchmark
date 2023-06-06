# Musae
## Musae-Github
**Description**: The GitHub Web and ML Developers dataset introduced in [MUSAE](https://arxiv.org/abs/1909.13021).
    Nodes represent developers on GitHub and edges are mutual follower relationships. Hyperedges are mutually following
    developer groups that contain at least 3 developers (i.e., maximal cliques with sizes of at least 3).
    It contains 37,300 nodes, 578,006 edges, 223,672 hyperedges, 128 (if the MUSAE preprocessed node embeddings
    are used) or 4,005 (if the raw node features are used) node features, and 4 classes.

### Access
**Python (Recommended)**
  ```python
    from hg.datasets import GitHub
    dataset = GitHub(root='/tmp/datasets/github')
    print(dataset[0])
    #Data(x=[37700, 128], edge_index=[2, 578006], y=[37700], hyperedge_index=[2, 1026826], num_hyperedges=223672)
  ```
Download Raw Data in JSON: [Zenodo](https://zenodo.org/record/7982540/files/musae_Github.json?download=1)
### References
- [Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)
### License: GNU General Public License v3.0

## Musae-Facebook
**Description**: The Facebook page-page network dataset introduced in the [MUSAE](https://arxiv.org/abs/1909.13021).
    Nodes represent verified pages on Facebook and edges are mutual likes. Hyperedges are mutually liked page groups
    that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3).
    It contains 22,470 nodes, 342,004 edges, 236,663 hyperedges, 128 (if the MUSAE preprocessed node embeddings
    are used) or 4,714 (if the raw node features are used) node features, and 4 classes

### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Facebook
    dataset = GitHub(root='/tmp/datasets/facebook')
    print(dataset[0])
    #Data(x=[37700, 128], edge_index=[2, 578006], y=[37700], hyperedge_index=[2, 1026826], num_hyperedges=223672)
  ```
Download Raw Data in JSON: [Zenodo](https://zenodo.org/record/7982540/files/musae_Facebook.json?download=1)
### References
- [Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)
### License: GNU General Public License v3.0


## Musae-Twitch

## Musae-Wiki

# Grand
## Grand-Tissues

Description of Dataset 2.

| Feature 1 | Feature 2 | Feature 3 |
|-----------|-----------|-----------|
| Value     | Value     | Value     |

Statistics


| Name| #Nodes | #Edges |
|-----|--------|--------|
|Value| Value  | Value  |

## Grand-Diseases

Description of Dataset 2.

| Feature 1 | Feature 2 | Feature 3 |
|-----------|-----------|-----------|
| Value     | Value     | Value     |

Statistics


| Name| #Nodes | #Edges |
|-----|--------|--------|
|Value| Value  | Value  |

# Amazon

Description of Dataset 2.

| Feature 1 | Feature 2 | Feature 3 |
|-----------|-----------|-----------|
| Value     | Value     | Value     |

Statistics


| Name| #Nodes | #Edges |
|-----|--------|--------|
|Value| Value  | Value  |

