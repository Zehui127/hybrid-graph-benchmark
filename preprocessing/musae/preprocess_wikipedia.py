import csv
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.utils import to_dense_adj, to_networkx


def find_max_cliques(name: str):
    graph_pyg = WikipediaNetwork(root='data/wikipedia/pyg-wikipedia', name=name, geom_gcn_preprocess=False)[0]
    x = graph_pyg.x

    edge_index_musae = pd.read_csv(f'data/wikipedia/musae-wikipedia/{name}/musae_{name}_edges.csv')
    edge_index = np.array([pd.concat([edge_index_musae['id1'], edge_index_musae['id2']]).values,
                           pd.concat([edge_index_musae['id2'], edge_index_musae['id1']]).values])
    edge_index = torch.Tensor(edge_index).long()

    y_musae = pd.read_csv(f'data/wikipedia/musae-wikipedia/{name}/musae_{name}_target.csv')
    y = torch.Tensor(y_musae['target'].values)

    graph = Data(x=x, edge_index=edge_index, y=y)
    graph_networkx = to_networkx(graph, to_undirected=True)
    max_cliques = list(nx.find_cliques(graph_networkx))
    max_cliques_processed = [clique for clique in max_cliques if len(clique) > 2]

    with open(f'data/wikipedia/wikipedia_{name}_cliques.pickle', 'wb') as f:
        pickle.dump(list(max_cliques), f)
    with open(f'data/wikipedia/wikipedia_{name}_cliques_processed.pickle', 'wb') as f:
        pickle.dump(list(max_cliques_processed), f)

    print('Total number of maximal cliques (including 2-cliques):', len(max_cliques))
    print('Total number of maximal cliques (excluding 2-cliques):', len(max_cliques_processed))
    print('Largest maximal clique size:', len(max(max_cliques_processed, key=len)))


def build_hyperedges(name: str):
    with open(f'data/wikipedia/wikipedia_{name}_cliques_processed.pickle', 'rb') as f:
        max_cliques = pickle.load(f)

    # with open(f'data/wikipedia/wikipedia_{name}_hyperedges.csv', 'w', newline='') as csvfile:
    with open(f'hyperedges/wikipedia_{name}_hyperedges.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['node', 'hyperedge'])
        writer.writeheader()
        for i, clique in enumerate(max_cliques):
            for node in clique:
                writer.writerow({'node': node, 'hyperedge': i})


if __name__ == '__main__':
    for name in ['chameleon', 'crocodile', 'squirrel']:
        print(f'Processing {name} dataset:')
        find_max_cliques(name)
        build_hyperedges(name)
        print()
