import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


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


def create_point_clouds(x, k, threshold):
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
    num_nodes = x.size(0)
    edge_distances = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)
    mask = edge_distances < threshold
    thresholded_edge_index = edge_index[:, mask]

    point_clouds = []
    for node in range(num_nodes):
        neighbors = thresholded_edge_index[1][thresholded_edge_index[0] == node]
        point_cloud = x[neighbors]
        point_clouds.append(point_cloud)

    return point_clouds


x = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float)

# Compute pairwise distances and create point clouds
k = 3  # Number of nearest neighbors to consider
threshold = 2.0  # Threshold value for edge distances
point_clouds = create_point_clouds(x, k, threshold)

# Print point clouds
for i, point_cloud in enumerate(point_clouds):
    print(f"Node {i}:")
    print(point_cloud)
