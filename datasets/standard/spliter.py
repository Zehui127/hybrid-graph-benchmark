import torch
import numpy as np
import math
import torch_geometric.utils as utils


def mask_split(dataset, original_mask=True,
               train_portion=0.6, eval_portion=0.2, test_portion=0.2):
    # re split to the train, eval and test mask to 60:20:20
    # only suppose to work with cora
    masks = []
    for data in dataset:
        if original_mask:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            num_nodes = data.num_nodes
            indexes = np.arange(num_nodes)
            np.random.shuffle(indexes)
            zeros = torch.zeros((num_nodes))

            train_point = math.ceil(num_nodes * train_portion)
            eval_point = math.ceil(num_nodes * (train_portion + eval_portion))

            train_mask = zeros.clone()
            val_mask = zeros.clone()
            test_mask = zeros.clone()
            train_mask[indexes[:train_point]] = 1
            val_mask[indexes[train_point:eval_point]] = 1
            test_mask[indexes[eval_point:]] = 1
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            train_mask = train_mask.type(torch.bool)
            val_mask = val_mask.type(torch.bool)
            test_mask = test_mask.type(torch.bool)

        masks.append((train_mask, val_mask, test_mask))
    return dataset, masks


def random_node_split(data, train_ratio=0.6, val_ratio=0.2, keep_joint_hyperedges=True):
    num_nodes = data.num_nodes
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.zeros(num_nodes)

    perm = torch.randperm(num_nodes)
    train_mask[perm[:num_train]] = 1
    val_mask[perm[num_train:num_train + num_val]] = 1
    test_mask[perm[num_train + num_val:]] = 1

    data.train_mask = train_mask.type(torch.bool)
    data.val_mask = val_mask.type(torch.bool)
    data.test_mask = test_mask.type(torch.bool)

    if keep_joint_hyperedges:
        # If keep_joint_hyperedges is set to True, then a hyperedge is included in a split dataset
        # as long as it contains a node in that split dataset.
        data.train_hyperedge_index = data.hyperedge_index[:, data.train_mask[data.hyperedge_index[0, :]]]
        data.val_hyperedge_index = data.hyperedge_index[:, data.val_mask[data.hyperedge_index[0, :]]]
        data.test_hyperedge_index = data.hyperedge_index[:, data.test_mask[data.hyperedge_index[0, :]]]
        data.num_train_hyperedges = data.train_hyperedge_index[1].unique().numel()
        data.num_val_hyperedges = data.val_hyperedge_index[1].unique().numel()
        data.num_test_hyperedges = data.test_hyperedge_index[1].unique().numel()
    else:
        # If keep_joint_hyperedges is set to False, then a hyperedge is included in a split dataset
        # only if every node in that hyperedge belong to that split dataset. This would drastically
        # reduce the number of total hyperedges in all splits.
        hyperedge_index = data.hyperedge_index
        train_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[:num_train])
        val_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[num_train:num_train + num_val])
        test_node_in_hyperedge_mask = torch.isin(hyperedge_index[0, :], perm[num_train + num_val:])

        train_hyperedge_to_exclude_mask = hyperedge_index[1, ~train_node_in_hyperedge_mask].unique()
        val_hyperedge_to_exclude_mask = hyperedge_index[1, ~val_node_in_hyperedge_mask].unique()
        test_hyperedge_to_exclude_mask = hyperedge_index[1, ~test_node_in_hyperedge_mask].unique()

        train_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], train_hyperedge_to_exclude_mask)
        val_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], val_hyperedge_to_exclude_mask)
        test_hyperedge_mask = ~torch.isin(hyperedge_index[1, :], test_hyperedge_to_exclude_mask)

        data.train_hyperedge_index = hyperedge_index[:, train_hyperedge_mask]
        data.val_hyperedge_index = hyperedge_index[:, val_hyperedge_mask]
        data.test_hyperedge_index = hyperedge_index[:, test_hyperedge_mask]
        data.num_train_hyperedges = data.num_hyperedges - train_hyperedge_to_exclude_mask.numel()
        data.num_val_hyperedges = data.num_hyperedges - val_hyperedge_to_exclude_mask.numel()
        data.num_test_hyperedges = data.num_hyperedges - test_hyperedge_to_exclude_mask.numel()

    return data

def create_edge_label(datasets,train_ratio=0.6, val_ratio=0.2):
        r"""This is used for edge prediction tasks
        Creates edge labels :obj:`data.edge_label` based on node labels
        :obj:`data.y`.

        Args:
            data (torch_geometric.data.Data): The graph data object.

        :rtype: :class:`torch_geometric.data.Data`
        """
        def edge_label(data):
            # we split the edge_index with all mode
            # get the edge index for train_message_ps = eval_mp,
            # train_supervision + eval_supervision + test_supervision = whole graph
            # test_mp = train_supervision + eval_supervision
            # data.edge_index = utils.to_undirected(data.edge_index)
            train_pos_edge_index, val_pos_edge_index, test_pos_edge_index  = random_edge_split(data,train_ratio, val_ratio)
            data.train_edge_index = data.val_edge_index = train_pos_edge_index
            data.test_edge_index = torch.cat([train_pos_edge_index, val_pos_edge_index], dim=1)
            data.train_label, data.train_edge_label_index = helper(train_pos_edge_index)
            data.val_label, data.val_edge_label_index = helper(val_pos_edge_index)
            data.test_label, data.test_edge_label_index = helper(test_pos_edge_index)
            return data
        def helper(edge_index):
            neg_edge_index = utils.negative_sampling(edge_index)
            # create edge_label
            pos_label = torch.ones(edge_index.size(1), dtype=torch.int64)
            neg_label = torch.zeros(neg_edge_index.size(1), dtype=torch.int64)
            edge_label = torch.cat([pos_label, neg_label], dim=0)
            return edge_label, torch.cat([edge_index, neg_edge_index], dim=1)
        return [edge_label(data) for data in datasets]

def random_edge_split(data, train_ratio=0.6, val_ratio=0.2):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        num_train = int(train_ratio * num_edges)
        num_val = int(val_ratio * num_edges)
        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[:num_train]]
        val_edges = edge_index[:, perm[num_train:num_train + num_val]]
        test_edges = edge_index[:, perm[num_train + num_val:]]
        return train_edges, val_edges, test_edges
