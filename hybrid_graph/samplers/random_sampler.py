import torch


class RandomNodeSampler(torch.utils.data.DataLoader):
    # Graphs sampled using this method can be very sparse.
    def __init__(self, data, num_samples, sample_size, mask=None, **kwargs):
        # Remove for PyTorch Lightning:
        # TODO: Copied from torch_geometric.loader.graph_saint.GraphSAINTSampler. Not sure if it is needed
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.data = data
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.mask = mask

        super().__init__(self, collate_fn=self._collate, **kwargs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # Apply the mask if provided
        if self.mask is not None:
            masked_nodes = torch.arange(self.data.num_nodes)[self.mask]
        else:
            masked_nodes = torch.arange(self.data.num_nodes)

        # Sample random nodes
        perm = torch.randperm(masked_nodes.size(0))
        shuffled_nodes = masked_nodes[perm]
        sampled_nodes = shuffled_nodes[:self.sample_size]

        # Get the subgraph induced by the sampled nodes
        x_sampled = self.data.x[sampled_nodes]
        y_sampled = self.data.y[sampled_nodes]

        # Get edges and hyperedges between the sampled nodes
        edge_mask = torch.isin(self.data.edge_index[0], sampled_nodes) & \
                    torch.isin(self.data.edge_index[1], sampled_nodes)
        edge_index_sampled = self.data.edge_index[:, edge_mask]
        hyperedge_mask = torch.isin(self.data.hyperedge_index[0], sampled_nodes)
        hyperedge_index_sampled = self.data.hyperedge_index[:, hyperedge_mask]
        _, inverse_indices = torch.unique(hyperedge_index_sampled[1], return_inverse=True)
        hyperedge_index_sampled[1] = inverse_indices

        # Remap node indices to the range [0, sample_size - 1]
        node_map = {node: i for i, node in enumerate(sampled_nodes.tolist())}
        edge_index_sampled.apply_(lambda node: node_map[node])
        hyperedge_index_sampled[0].apply_(lambda node: node_map[node])

        return x_sampled, y_sampled, edge_index_sampled, hyperedge_index_sampled

    def _collate(self, data_list):
        assert len(data_list) == 1
        x_sampled, y_sampled, edge_index_sampled, hyperedge_index_sampled = data_list[0]

        data = self.data.__class__()
        data.num_nodes = x_sampled.size(0)
        data.x = x_sampled
        data.y = y_sampled
        data.edge_index = edge_index_sampled
        data.hyperedge_index = hyperedge_index_sampled
        data.num_hyperedges = hyperedge_index_sampled[1, -1] + 1

        return data


class RandomHyperedgeSampler(torch.utils.data.DataLoader):
    # Graphs sampled using this method can be dense.
    # WARNING: THIS METHOD CAN GET VERY SLOW FOR LARGE SAMPLES.
    def __init__(self, data, num_samples, min_sample_size, mask=None, **kwargs):
        # Remove for PyTorch Lightning:
        # TODO: Copied from torch_geometric.loader.graph_saint.GraphSAINTSampler. Not sure if it is needed
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.data = data
        self.num_samples = num_samples
        self.min_sample_size = min_sample_size
        self.mask = mask

        super().__init__(self, collate_fn=self._collate, **kwargs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # Sample hyperedges until they contain at least min_sample_size nodes, and apply the mask if provided
        perm = torch.randperm(self.data.num_hyperedges)
        sampled_nodes_mask = torch.zeros(self.data.num_nodes)
        i = 0
        while int(torch.sum(sampled_nodes_mask)) < self.min_sample_size:
            sample = perm[i]
            sampled_nodes_mask[self.data.hyperedge_index[0, self.data.hyperedge_index[1, :] == sample]] = 1
            if self.mask is not None:
                sampled_nodes_mask = sampled_nodes_mask & self.mask
            i += 1
        sampled_nodes = torch.arange(self.data.num_nodes)[sampled_nodes_mask.type(torch.bool)]

        # Get the subgraph induced by the sampled nodes
        x_sampled = self.data.x[sampled_nodes]
        y_sampled = self.data.y[sampled_nodes]

        # Get edges and hyperedges between the sampled nodes
        edge_mask = torch.isin(self.data.edge_index[0], sampled_nodes) & \
                    torch.isin(self.data.edge_index[1], sampled_nodes)
        edge_index_sampled = self.data.edge_index[:, edge_mask]
        hyperedge_mask = torch.isin(self.data.hyperedge_index[0], sampled_nodes)
        hyperedge_index_sampled = self.data.hyperedge_index[:, hyperedge_mask]
        _, inverse_indices = torch.unique(hyperedge_index_sampled[1], return_inverse=True)
        hyperedge_index_sampled[1] = inverse_indices

        # Remap node indices to the range [0, sample_size - 1]
        node_map = {node: i for i, node in enumerate(sampled_nodes.tolist())}
        edge_index_sampled.apply_(lambda node: node_map[node])
        hyperedge_index_sampled[0].apply_(lambda node: node_map[node])

        return x_sampled, y_sampled, edge_index_sampled, hyperedge_index_sampled

    def _collate(self, data_list):
        assert len(data_list) == 1
        x_sampled, y_sampled, edge_index_sampled, hyperedge_index_sampled = data_list[0]

        data = self.data.__class__()
        data.num_nodes = x_sampled.size(0)
        data.x = x_sampled
        data.y = y_sampled
        data.edge_index = edge_index_sampled
        data.hyperedge_index = hyperedge_index_sampled
        data.num_hyperedges = hyperedge_index_sampled[1, -1] + 1

        return data
