import csv
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.utils import to_networkx

NUM_NODES = {
    'chameleon': 2277,
    'crocodile': 11631,
    'squirrel': 5201
}


def find_max_cliques(name: str):
    graph_pyg = WikipediaNetwork(root='data/wikipedia/pyg-wikipedia', name=name, geom_gcn_preprocess=False)[0]
    x = graph_pyg.x

    edge_index_musae = pd.read_csv(f'data/wikipedia/musae-wikipedia/{name}/musae_{name}_edges.csv')
    edge_index = np.array([pd.concat([edge_index_musae['id1'], edge_index_musae['id2']]).values,
                           pd.concat([edge_index_musae['id2'], edge_index_musae['id1']]).values])
    edge_index = np.unique(edge_index, axis=1)
    mask = edge_index[0, :] != edge_index[1, :]
    edge_index = edge_index[:, mask]
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


def save_processed_data(name: str):
    pyg_data = np.load(f'data/wikipedia/pyg-wikipedia/{name}/raw/{name}.npz', 'r', allow_pickle=True)
    emb_features = pyg_data['features']
    target = pyg_data['target']

    with open(f'data/wikipedia/musae-wikipedia/{name}/musae_{name}_features.json', 'r') as f:
        raw_featrues_json = json.load(f)

    max_feature_idx = -1
    for features in raw_featrues_json.values():
        max_feature_idx = max(max_feature_idx, max(features))

    print(name, 'num features:', max_feature_idx + 1)
    raw_features = np.zeros((NUM_NODES[name], max_feature_idx + 1))
    for node, features in raw_featrues_json.items():
        for feature_idx in features:
            raw_features[int(node), feature_idx] = 1

    edges_df = pd.read_csv(f'data/wikipedia/musae-wikipedia/{name}/musae_{name}_edges.csv')
    edges = np.array([pd.concat([edges_df['id1'], edges_df['id2']]).values,
                      pd.concat([edges_df['id2'], edges_df['id1']]).values])
    edges = np.unique(edges, axis=1)
    mask = edges[0, :] != edges[1, :]
    edges = edges[:, mask]

    hyperedges_df = pd.read_csv(f'hyperedges/wikipedia_{name}_hyperedges.csv')
    hyperedges = np.array([hyperedges_df['node'].values, hyperedges_df['hyperedge'].values])

    np.savez_compressed(f'data/wikipedia/wikipedia_{name}.npz',
                        features=emb_features, raw_features=raw_features,
                        target=target, edges=edges, hyperedges=hyperedges)


if __name__ == '__main__':
    for name in ['chameleon', 'crocodile', 'squirrel']:
        print(f'Processing {name} dataset:')
        find_max_cliques(name)
        build_hyperedges(name)
        save_processed_data(name)
        print()
