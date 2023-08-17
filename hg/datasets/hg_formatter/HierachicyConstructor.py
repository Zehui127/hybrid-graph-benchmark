import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh


class HierachicyConstructor:
    def __init__(self, graph, max_cluster_size, num_clusters):
        # convert torch geometric graph to networkx graph
        self.graph = graph
        self.max_cluster_size = max_cluster_size
        self.num_clusters = num_clusters
        self.nx_graph = self.convert_to_nx_graph(graph)

    def construct(self, type='greedy'):
        if type == 'greedy':
            self.graph.clusters = self.greedy_partition_corrected(
                             self.nx_graph, self.max_cluster_size)
        elif type == 'spectral':
            self.graph.clusters = self.spectral_clustering(
                                  self.nx_graph, self.num_clusters)
        else:
            raise NotImplementedError
        self.graph.incidence_matrix = self.compute_incidence_matrix(
                                      self.nx_graph, self.graph.clusters)
        return self.graph

    def convert_to_nx_graph(self, graph):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(graph.num_nodes))
        nx_graph.add_edges_from(graph.edge_index.t().tolist())
        return nx_graph

    def greedy_partition_corrected(self, graph, max_cluster_size):
        """
        Greedy graph partitioning algorithm with non-overlapping clusters.

        Parameters:
        - graph: A NetworkX graph.
        - max_cluster_size: Maximum number of nodes in a cluster.

        Returns:
        A list of clusters, each cluster is a list of nodes.
        """
        nodes_to_visit = set(graph.nodes())  # This set ensures we keep track of unvisited nodes
        clusters = []

        while nodes_to_visit:
            start_node = nodes_to_visit.pop()
            cluster = [start_node]
            frontier = [neigh for neigh in graph.neighbors(start_node) if neigh in nodes_to_visit]

            while frontier and len(cluster) < max_cluster_size:
                node = max(frontier, key=lambda n: graph.subgraph(frontier).degree(n))
                cluster.append(node)
                frontier.extend([neigh for neigh in graph.neighbors(node) if neigh in nodes_to_visit and neigh not in frontier])
                frontier.remove(node)
                nodes_to_visit.discard(node)  # Ensure nodes are removed from nodes_to_visit once they're added to a cluster
            clusters.append(cluster)
        print("Finished greedy partitioning")
        return clusters

    def compute_incidence_matrix(self, graph, clusters):
        """
        Compute the incidence matrix for clusters.

        Parameters:
        - graph: A NetworkX graph.
        - clusters: List of clusters.

        Returns:
        A matrix where entry (i, j) indicates the number of edges connecting cluster i to cluster j.
        """
        num_clusters = len(clusters)
        incidence_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

        for i in range(num_clusters):
            for j in range(num_clusters):
                if i != j:
                    edges_between_clusters = len(list(nx.edge_boundary(graph, clusters[i], clusters[j])))
                    incidence_matrix[i, j] = edges_between_clusters

        return incidence_matrix

    def spectral_clustering(self, graph, num_clusters):
        """
        Spectral clustering algorithm.

        Parameters:
        - graph: A NetworkX graph.
        - num_clusters: Number of desired clusters.

        Returns:
        A list of clusters, each cluster is a list of nodes.
        """
        # Construct the adjacency matrix
        A = nx.adjacency_matrix(graph).astype(float)

        # Construct the degree matrix
        degrees = np.array(graph.degree())[:, 1]
        D = np.diag(degrees)

        # Compute the normalized Laplacian matrix
        L = D - A
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Compute the eigenvectors of the normalized Laplacian
        eigenvalues, eigenvectors = eigsh(L_norm, k=num_clusters, which='SM')

        # Use the eigenvectors for clustering nodes
        U = eigenvectors
        rows_norm = np.linalg.norm(U, axis=1, keepdims=True)
        U /= rows_norm

        # Apply k-means clustering on the rows of U
        kmeans = KMeans(n_clusters=num_clusters).fit(U)
        labels = kmeans.labels_

        # Organize nodes into clusters
        clusters = [[] for _ in range(num_clusters)]
        for node, label in enumerate(labels):
            clusters[label].append(node)

        return clusters
