from typing import Optional

import torch
from torch_geometric.loader import GraphSAINTSampler


class HypergraphSAINTSampler(GraphSAINTSampler):
    # def _collate(self, data_list):  # torch-geometric>=2.3.0
    def __collate__(self, data_list):  # torch-geometric==2.2.0
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        hyperedge_mask = torch.isin(self.data.hyperedge_index[0], node_idx)
        data.hyperedge_index = self.data.hyperedge_index[:, hyperedge_mask]
        _, inverse_indices = torch.unique(data.hyperedge_index[1], return_inverse=True)
        data.hyperedge_index[1] = inverse_indices
        data.num_hyperedges = data.hyperedge_index[1, -1] + 1

        # Remap node indices in hyperedge_index to the range [0, sample_size - 1]
        node_map = {node: i for i, node in enumerate(node_idx.tolist())}
        data.hyperedge_index[0].apply_(lambda node: node_map[node])

        for key, item in self.data:
            if key in ['edge_index', 'hyperedge_index', 'num_nodes', 'num_hyperedges']:
                continue
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data


class HypergraphSAINTNodeSampler(HypergraphSAINTSampler):
    # def _sample_nodes(self, batch_size):  # torch-geometric>=2.3.0
    def __sample_nodes__(self, batch_size):  # torch-geometric==2.2.0
        edge_sample = torch.randint(0, self.E, (batch_size, self.batch_size), dtype=torch.long)

        return self.adj.storage.row()[edge_sample]


class HypergraphSAINTEdgeSampler(HypergraphSAINTSampler):
    # def _sample_nodes(self, batch_size):  # torch-geometric>=2.3.0
    def __sample_nodes__(self, batch_size):  # torch-geometric==2.2.0
        row, col, _ = self.adj.coo()

        deg_in = 1. / self.adj.storage.colcount()
        deg_out = 1. / self.adj.storage.rowcount()
        prob = (1. / deg_in[row]) + (1. / deg_out[col])

        # Parallel multinomial sampling (without replacement)
        # https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
        rand = torch.rand(batch_size, self.E).log() / (prob + 1e-10)
        edge_sample = rand.topk(self.batch_size, dim=-1).indices

        source_node_sample = col[edge_sample]
        target_node_sample = row[edge_sample]

        return torch.cat([source_node_sample, target_node_sample], -1)


class HypergraphSAINTRandomWalkSampler(HypergraphSAINTSampler):
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)

    @property
    # def _filename(self):  # torch-geometric>=2.3.0
    def __filename__(self):  # torch-geometric==2.2.0
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    # def _sample_nodes(self, batch_size):  # torch-geometric>=2.3.0
    def __sample_nodes__(self, batch_size):  # torch-geometric==2.2.0
        start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)
