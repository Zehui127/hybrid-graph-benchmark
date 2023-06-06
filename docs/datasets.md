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



## Musae-Twitch

**Description**: The Facebook page-page network dataset introduced in the [MUSAE](https://arxiv.org/abs/1909.13021).
    Nodes represent gamers on Twitch and edges are followerships between them. Hyperedges are mutually following
    user groups that contain at least 3 gamers (i.e., maximal cliques with sizes of at least 3).
    The task is to predict whether a user streams mature content.
    Each dataset contains 128 (if the MUSAE preprocessed node embeddings are used) or 4,005 (if the raw node features
    are used) node features, and 2 classes.

### Graphs

| Name | #nodes |  #edges | #hyperedges | #features   | #classes |
|------|-------:|-------:|------------:|------------:|--------:|
| DE   |  9,498 | 306,276 |     297,315 | 128 or 3,170 |       2 |
| EN   |  7,126 |  70,648 |      13,248 | 128 or 3,170 |       2 |
| ES   |  4,648 | 118,764 |      77,135 | 128 or 3,170 |       2 |
| FR   |  6,549 | 225,332 |     172,653 | 128 or 3,170 |       2 |
| PT   |  1,912 |  62,598 |      74,830 | 128 or 3,170 |       2 |
| RU   |  4,385 |  74,608 |      25,673 | 128 or 3,170 |       2 |

### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Twitch
    # for other languages, use name='EN', 'ES', 'FR', 'PT', 'RU'
    dataset = Twitch('/tmp/datasets/twitch',name='DE') 
    print(dataset[0])
    # Data(x=[9498, 128], edge_index=[2, 306276], y=[9498], hyperedge_index=[2, 2277625], num_hyperedges=297315)
  ```
Download Raw Data in JSON: 
[Zenodo(DE)](https://zenodo.org/record/7982540/files/musae_Twitch_DE.json?download=1) 
[Zenodo(EN)](https://zenodo.org/record/7982540/files/musae_Twitch_EN.json?download=1)
[Zenodo(ES)](https://zenodo.org/record/7982540/files/musae_Twitch_ES.json?download=1)
[Zenodo(FR)](https://zenodo.org/record/7982540/files/musae_Twitch_FR.json?download=1)
[Zenodo(PT)](https://zenodo.org/record/7982540/files/musae_Twitch_PT.json?download=1)
[Zenodo(RU)](https://zenodo.org/record/7982540/files/musae_Twitch_RU.json?download=1)


## Musae-Wiki

**Description**: The Wikipedia network dataset introduced in the [MUSAE](https://arxiv.org/abs/1909.13021). 
    Nodes represent web pages and edges represent mutual hyperlinks between them. Hyperedges are mutually linked
    page groups that contain at least 3 pages (i.e., maximal cliques with sizes of at least 3).
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average monthly traffic of the web page.

### Graphs

| Name      | #nodes |  #edges | #hyperedges |    #features |
|-----------|-------:|-------:|------------:|-------------:|
| chameleon |  2,277 |  62,742 |      14,650 | 128 or 3,132 |
| crocodile | 11,631 | 341,546 |     121,431 | 128 or 13,183 |
| squirrel  |  5,201 | 396,706 |     220,678 | 128 or 3,148 |


### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Wikipedia
    # for other two graphs, use name='crocodile', 'squirrel'
    dataset = Wikipedia('/tmp/datasets/wiki',name='chameleon') 
    print(dataset[0])
    #Data(x=[2277, 128], edge_index=[2, 62742], y=[2277], hyperedge_index=[2, 113444], num_hyperedges=14650)
  ```
Download Raw Data in JSON: 
[Zenodo(chameleon)](https://zenodo.org/record/7982540/files/musae_Wikipedia_chameleon.json?download=1)
[Zenodo(crocodile)](https://zenodo.org/record/7982540/files/musae_Wikipedia_crocodile.json?download=1)
[Zenodo(squirrel)](https://zenodo.org/record/7982540/files/musae_Wikipedia_squirrel.json?download=1)
## References
- [Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)

## License: GNU General Public License v3.0


# Grand
## Grand-Tissues

**Description** We select and build six gene regulatory networks in different tissues from
[GRAND](https://academic.oup.com/nar/article/50/D1/D610/6368528), a public database for gene regulation. 
Nodes represent gene regulatory elements with three distinct types: protein-encoding gene, lncRNA
gene, and other regulatory elements. Edges are regulatory effects between genes.
The task is a multi-class classification of gene regulatory elements. We train a [CNN](https://www.nature.com/articles/s41576-019-0122-6) and use it to take the gene sequence as input and create a 4,651-dimensional embedding for each
node. The hyperedges are constructed by grouping nearby genomic elements on the chromosomes,
i.e., the genomic elements within 200k base pair distance are grouped as hyperedges.

### Graphs

| Name                            | #nodes | #edges | #hyperedges | #features | #classes |
|---------------------------------|-------:|-------:|------------:|----------:|---------:|
| Brain                           |  6,196 |  6,245 |      11,878 |      4608 |        3 |
| Lung                            |  6,119 |  6,160 |      11,760 |      4608 |        3 |
| Breast                          |  5,921 |  5,910 |      11,400 |      4608 |        3 |
| Artery_Coronary                 |  5,755 |  5,722 |      11,222 |      4608 |        3 |
| Artery_Aorta                    |  5,848 |  5,823 |      11,368 |      4608 |        3 |
| Stomach                         |  5,745 |  5,694 |      11,201 |      4608 |        3 |

### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Grand
    # for other tissues, use name='Lung', 'Breast', 'Artery_Coronary', 'Artery_Aorta', 'Stomach' 
    dataset = Grand('/tmp/datasets/grand',name='Brain') 
    print(dataset[0])
    #Data(x=[6196, 4608], edge_index=[2, 6245], y=[6196], hyperedge_index=[2, 15388], num_hyperedges=11878)
  ```
Download Raw Data in JSON:
[Zenodo(Brain)](https://zenodo.org/record/7982540/files/grand_Brain.json?download=1)
[Zenodo(Lung)](https://zenodo.org/record/7982540/files/grand_Lung.json?download=1)
[Zenodo(Breast)](https://zenodo.org/record/7982540/files/grand_Breast.json?download=1)
[Zenodo(Artery_Coronary)](https://zenodo.org/record/7982540/files/grand_ArteryCoronary.json?download=1)
[Zenodo(Artery_Aorta)](https://zenodo.org/record/7982540/files/grand_ArteryAorta.json?download=1)
[Zenodo(Stomach)](https://zenodo.org/record/7982540/files/grand_Stomach.json?download=1)


## Grand-Diseases

**Description** We select and build four gene regulatory networks in different genetic diseases from
[GRAND](https://academic.oup.com/nar/article/50/D1/D610/6368528), a public database for gene regulation. 
Nodes represent gene regulatory elements with three distinct types: protein-encoding gene, lncRNA
gene, and other regulatory elements. Edges are regulatory effects between genes.
The task is a multi-class classification of gene regulatory elements. We train a [CNN](https://www.nature.com/articles/s41576-019-0122-6) and use it to take the gene sequence as input and create a 4,651-dimensional embedding for each
node. The hyperedges are constructed by grouping nearby genomic elements on the chromosomes,
i.e., the genomic elements within 200k base pair distance are grouped as hyperedges.

### Graphs

| Name                            | #nodes | #edges | #hyperedges | #features | #classes |
|---------------------------------|-------:|-------:|------------:|----------:|---------:|
| Leukemia                        |  4,651 |  6,362 |       7,812 |      4608 |        3 |
| Kidney_renal_papillary_cell_carcinoma |  4,319 |  5,599 |       7,369 |      4608 |        3 |
| Lung_cancer                     |  4,896 |  6,995 |       8,179 |      4608 |        3 |
| Stomach_cancer                  |  4,518 |  6,051 |       7,611 |      4608 |        3 |

### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Grand
    # for other diseases, use name='Kidney_renal_papillary_cell_carcinoma', 'Lung_cancer', 'Stomach_cancer'
    dataset = Grand('/tmp/datasets/grand',name='Leukemia') 
    print(dataset[0])
    #Data(x=[4651, 4608], edge_index=[2, 6362], y=[4651], hyperedge_index=[2, 10346], num_hyperedges=7812)
  ```
Download Raw Data in JSON:
[Zenodo(Leukemia)](https://zenodo.org/record/7982540/files/grand_Leukemia.json?download=1)
[Zenodo(KidneyCancer)](https://zenodo.org/record/7982540/files/grand_KidneyCancer.json?download=1)
[Zenodo(LungCancer)](https://zenodo.org/record/7982540/files/grand_Lungcancer.json?download=1)
[Zenodo(StomachCancer)](https://zenodo.org/record/7982540/files/grand_Stomachcancer.json?download=1)


## References
- [GRAND: a database of gene regulatory network models across human conditions](https://academic.oup.com/nar/article/50/D1/D610/6368528)
- [Deep learning: new computational modelling techniques for genomics](https://www.nature.com/articles/s41576-019-0122-6)

## License: GNU General Public License v3.0

# Amazon

## Computers & Photos

**Description**: Amazon Computers and Photos are built two e-commerce hypergraph datasets
based on the [Amazon Product Reviews dataset](https://arxiv.org/abs/2103.00020).
Nodes represent products, and an edge between two products is established if a user
buys these two products or writes reviews for both. However, unlike those existing datasets, we
introduce the image modality into the construction of hyperedge. To be specific, the raw images are
fed into a [CLIP](https://arxiv.org/abs/1506.04757) classifier, and a 512-dimensional feature embedding for each
image is returned to assist the clustering. The hyperedges are then constructed by grouping products
whose image embeddings’ pairwise distances are within a certain threshold.

### Graphs

| Name      | #nodes  | #edges  | #hyperedges | #features | #classes |
|-----------|--------:|--------:|------------:|----------:|---------:|
| Photos    |   6,777 |  45,306 |       6,777 |      1000 |       10 |
| Computers |  10,226 |  55,324 |      10,226 |      1000 |       10 |


### Access
**Python (Recommended)**
  ```python
    from hg.datasets import Amazon
    # for other datasets, use name='Photos'
    dataset = Amazon('/tmp/datasets/amazon',name='Computers')
    print(dataset[0])
    # Data(x=[10226, 1000], edge_index=[2, 55324], y=[10226], hyperedge_index=[2, 40903], num_hyperedges=10226)
  ```

## References
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Image-based Recommendations on Styles and Substitutes](https://arxiv.org/abs/1506.04757)

## License: [Amazon Service licence](https://s3.amazonaws.com/amazon-reviews-pds/LICENSE.txt)