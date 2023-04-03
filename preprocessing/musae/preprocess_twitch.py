import csv
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import Twitch
from torch_geometric.utils import to_dense_adj, to_networkx

NUM_NODES = {
    'DE': 9498,
    'EN': 7126,
    'ES': 4648,
    'FR': 6549,
    'PT': 1912,
    'RU': 4385
}
NUM_RAW_FEATURES = 3170


def validate_dataset(name: str):
    graph_pyg = Twitch(root='data/twitch/pyg-twitch', name=name[:2])[0]
    adj_pyg = to_dense_adj(graph_pyg.edge_index)
    adj_pyg -= torch.eye(graph_pyg.num_nodes)
    y_pyg = graph_pyg.y

    edge_index_musae = pd.read_csv(f'data/twitch/musae-twitch/{name}/musae_{name}_edges.csv')
    adj_musae = np.array([pd.concat([edge_index_musae['from'], edge_index_musae['to']]).values,
                          pd.concat([edge_index_musae['to'], edge_index_musae['from']]).values])
    adj_musae = torch.Tensor(adj_musae).long()
    adj_musae = to_dense_adj(adj_musae)

    y_musae = pd.read_csv(f'data/twitch/musae-twitch/{name}/musae_{name}_target.csv', index_col='new_id')
    y_musae.sort_index(inplace=True)
    y_musae = torch.Tensor(y_musae['mature'].values)

    if name == 'FR':
        assert torch.equal(adj_pyg[:, :NUM_NODES['FR'], :NUM_NODES['FR']], adj_musae)
    else:
        assert torch.equal(adj_pyg, adj_musae)
    assert torch.equal(y_pyg, y_musae)


def find_max_cliques(name: str):
    graph = Twitch(root='data/twitch/pyg-twitch', name=name[:2])[0]
    graph_networkx = to_networkx(graph, to_undirected=True)
    max_cliques = list(nx.find_cliques(graph_networkx))
    max_cliques_processed = [clique for clique in max_cliques if len(clique) > 2]

    with open(f'data/twitch/twitch_{name[:2]}_cliques.pickle', 'wb') as f:
        pickle.dump(list(max_cliques), f)
    with open(f'data/twitch/twitch_{name[:2]}_cliques_processed.pickle', 'wb') as f:
        pickle.dump(list(max_cliques_processed), f)

    print('Total number of maximal cliques (including 2-cliques):', len(max_cliques))
    print('Total number of maximal cliques (excluding 2-cliques):', len(max_cliques_processed))
    print('Largest maximal clique size:', len(max(max_cliques_processed, key=len)))


def build_hyperedges(name: str):
    with open(f'data/twitch/twitch_{name[:2]}_cliques_processed.pickle', 'rb') as f:
        max_cliques = pickle.load(f)

    # with open(f'data/twitch/twitch_{name[:2]}_hyperedges.csv', 'w', newline='') as csvfile:
    with open(f'hyperedges/twitch_{name[:2]}_hyperedges.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['node', 'hyperedge'])
        writer.writeheader()
        for i, clique in enumerate(max_cliques):
            for node in clique:
                writer.writerow({'node': node, 'hyperedge': i})


def save_processed_data(name: str):
    pyg_data = np.load(f'data/twitch/pyg-twitch/{name[:2]}/raw/{name[:2]}.npz', 'r', allow_pickle=True)
    if name == 'FR':
        emb_features = pyg_data['features'][:NUM_NODES['FR']]
        target = pyg_data['target'][:NUM_NODES['FR']]
    else:
        emb_features = pyg_data['features']
        target = pyg_data['target']

    with open(f'data/twitch/musae-twitch/{name}/musae_{name}_features.json', 'r') as f:
        raw_featrues_json = json.load(f)

    raw_features = np.zeros((NUM_NODES[name[:2]], NUM_RAW_FEATURES))
    for node, features in raw_featrues_json.items():
        for feature_idx in features:
            raw_features[int(node), feature_idx] = 1

    edges_df = pd.read_csv(f'data/twitch/musae-twitch/{name}/musae_{name}_edges.csv')
    edges = np.array([pd.concat([edges_df['from'], edges_df['to']]).values,
                      pd.concat([edges_df['to'], edges_df['from']]).values])

    hyperedges_df = pd.read_csv(f'hyperedges/twitch_{name[:2]}_hyperedges.csv')
    hyperedges = np.array([hyperedges_df['node'].values, hyperedges_df['hyperedge'].values])

    np.savez_compressed(f'data/twitch/twitch_{name[:2]}.npz',
                        features=emb_features, raw_features=raw_features,
                        target=target, edges=edges, hyperedges=hyperedges)


if __name__ == '__main__':
    for name in ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']:
        print(f'Processing {name[:2]} dataset:')
        validate_dataset(name)
        find_max_cliques(name)
        build_hyperedges(name)
        save_processed_data(name)
        print()
