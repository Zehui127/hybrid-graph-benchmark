import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import degree


class GraphFormatter(object):
    """
    A class to create a hypergraph from a graph using a threshold value.
    based on the distance between the node embeddings.
    """

    def __init__(self, k=None, percentile=80):
        self.k = k
        self.percentile = percentile

    def __call__(self, data):
        self.k = self.k if self.k is not None else degree(data.edge_index[0]).float().mean().item()
        self.k = int(self.k)
        x, y = data.x, data.y
        if hasattr(data, "emb"):
            hyper_x = data.emb
        else:
            hyper_x = x
        hyper_edge_index = get_hyper_edge_index(hyper_x, self.k, self.percentile)

        new_data = Data(
            x=x,
            edge_index=data.edge_index,
            y=y,
            hyperedge_index=hyper_edge_index,
            num_hyperedges=hyper_edge_index.shape[1],
        )
        if hasattr(data, "train_mask"):
            new_data.train_mask = data.train_mask
        if hasattr(data, "val_mask"):
            new_data.val_mask = data.val_mask
        if hasattr(data, "test_mask"):
            new_data.test_mask = data.test_mask
        return new_data


def compute_pairwise_distances(x, k, loop=True):
    """
    Compute pairwise distances between all the nodes in the graph.

    Args:
        x (torch.Tensor): Node features of shape (num_nodes, num_features).
        k (int): The number of nearest neighbors to consider.
        loop (bool): Whether to include self-loops in the output graph.

    Returns:
        edge_index (torch.Tensor): The edge indices of the k-nearest neighbor graph of shape (2, num_edges).
    """
    edge_index = knn_graph(x, k=k, batch=None, loop=loop)
    return edge_index


def create_point_clouds(x, k, threshold=None):
    """
    Create point clouds using a threshold value.

    Args:
        x (torch.Tensor): Node features of shape (num_nodes, num_features).
        k (int): The number of nearest neighbors to consider.
        threshold (float): The threshold value for edge distances.

    Returns:
        point_clouds (list): A list of point clouds as torch.Tensor.
    """
    edge_index = compute_pairwise_distances(x, k)

    if threshold is not None:
        edge_distances = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)
        mask = edge_distances < threshold
        thresholded_edge_index = edge_index[:, mask]
    else:
        thresholded_edge_index = edge_index

    return thresholded_edge_index, edge_index

def find_threshold(x, k, percentile):
    """
    Find the threshold value based on a certain percentile of the distance distribution.

    Args:
        x (torch.Tensor): Node features of shape (num_nodes, num_features).
        k (int): The number of nearest neighbors to consider.
        percentile (float): The percentile value for the threshold.

    Returns:
        threshold (float): The threshold value.
    """
    edge_index = compute_pairwise_distances(x, k)
    edge_distances = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)
    q = percentile / 100
    threshold = torch.quantile(edge_distances, q)
    return threshold


def get_hyper_edge_index(x, k, percentile):
    threshold = find_threshold(x, k, percentile)
    edge_index, _ = create_point_clouds(x, k, threshold)
    return edge_index
