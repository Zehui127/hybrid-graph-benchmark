import csv
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.utils import to_dense_adj, to_networkx

NUM_NODES = 22470


def validate_dataset():
    graph_pyg = FacebookPagePage(root='data/facebook/pyg-facebook')[0]
    adj_pyg = to_dense_adj(graph_pyg.edge_index)
    y_pyg = graph_pyg.y

    edge_index_musae = pd.read_csv('data/facebook/musae-facebook/musae_facebook_edges.csv')
    adj_musae = np.array([pd.concat([edge_index_musae['id_1'], edge_index_musae['id_2']]).values,
                          pd.concat([edge_index_musae['id_2'], edge_index_musae['id_1']]).values])
    adj_musae = torch.Tensor(adj_musae).long()
    adj_musae = to_dense_adj(adj_musae)

    y_musae = pd.read_csv('data/facebook/musae-facebook/musae_facebook_target.csv')
    y_musae.replace({'page_type': {
        'tvshow': 0,
        'company': 1,
        'government': 2,
        'politician': 3
    }}, inplace=True)
    y_musae = torch.Tensor(y_musae['page_type'].values)

    assert torch.equal(adj_pyg, adj_musae)
    assert torch.equal(y_pyg, y_musae)


def find_max_cliques():
    graph = FacebookPagePage(root='data/facebook/pyg-facebook')[0]
    graph_networkx = to_networkx(graph, to_undirected=True)
    max_cliques = list(nx.find_cliques(graph_networkx))
    max_cliques_processed = [clique for clique in max_cliques if len(clique) > 2]

    with open('data/facebook/facebook_cliques.pickle', 'wb') as f:
        pickle.dump(list(max_cliques), f)
    with open('data/facebook/facebook_cliques_processed.pickle', 'wb') as f:
        pickle.dump(list(max_cliques_processed), f)

    print('Total number of maximal cliques (including 2-cliques):', len(max_cliques))
    print('Total number of maximal cliques (excluding 2-cliques):', len(max_cliques_processed))
    print('Largest maximal clique size:', len(max(max_cliques_processed, key=len)))


def build_hyperedges():
    with open('data/facebook/facebook_cliques_processed.pickle', 'rb') as f:
        max_cliques = pickle.load(f)

    # with open('data/facebook/facebook_hyperedges.csv', 'w', newline='') as csvfile:
    with open('hyperedges/facebook_hyperedges.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['node', 'hyperedge'])
        writer.writeheader()
        for i, clique in enumerate(max_cliques):
            for node in clique:
                writer.writerow({'node': node, 'hyperedge': i})


def save_processed_data():
    with open('data/facebook/musae-facebook/musae_facebook_features.json', 'r') as f:
        raw_featrues_json = json.load(f)

    max_feature_idx = -1
    for features in raw_featrues_json.values():
        max_feature_idx = max(max_feature_idx, max(features))

    raw_features = np.zeros((NUM_NODES, max_feature_idx + 1))
    for node, features in raw_featrues_json.items():
        for feature_idx in features:
            raw_features[int(node), feature_idx] = 1

    hyperedges_df = pd.read_csv('hyperedges/facebook_hyperedges.csv')
    hyperedges = np.array([hyperedges_df['node'].values, hyperedges_df['hyperedge'].values])

    np.savez_compressed('data/facebook/facebook_preprocessed.npz', raw_features=raw_features, hyperedges=hyperedges)


if __name__ == '__main__':
    validate_dataset()
    find_max_cliques()
    build_hyperedges()
    save_processed_data()