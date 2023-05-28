from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
class EmbeddingVisualizer:
    def __init__(self, data, title='Embedding Visualizer'):
        self.embedding = data.x
        self.labels = data.y
        self.title = title
        self.tsne = TSNE(n_components=2, random_state=0)
        self.tsne_obj = self.tsne.fit_transform(self.embedding)
    def plot(self):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=self.tsne_obj[:, 0],
            y=self.tsne_obj[:, 1],
            hue=self.labels,
            palette='Spectral',
            legend='full',
            s=60
        )
        plt.title(self.title)
        plt.savefig(self.title + ".png")
class HyperEdgeSizeHist:
    #TODO: need to be fixed
    def __init__(self, data, title='HyperEdge Size Histogram'):
        self.title = title
        self.counts = data.hyperedge_index[1].bincount()
    def plot(self):
        plt.figure(figsize=(10, 8))
        plt.hist(self.counts, log=True, alpha=0.5)
        plt.xlabel('HyperEdge Size')
        plt.ylabel('Count')
        plt.yscale('log')
        plt.title(self.title)
        plt.savefig(self.title + ".png")

class GraphStats:
    def __init__(self, data: Data):
        self.data = data
        print(self.data)
        self.graph = to_networkx(data, to_undirected=True) # Convert to a networkx graph for some computations
    def plot(self):
        #EmbeddingVisualizer(self.data).plot()
        HyperEdgeSizeHist(self.data).plot()
    def get_all_stats(self):
        HyperEdgeSizeHist(self.data).plot()
        return {
            'num_nodes': self.num_nodes(),
            'num_edges': self.num_edges(),
            'average_degree': self.average_degree(),
            'clustering_coefficient': self.clustering_coefficient(),
            'density': self.density(),
            'diameter': self.diameter(),
            'average_size_of_hyperedge': self.average_size_of_hyperedge()
        }
    def average_size_of_hyperedge(self):
        return torch.sum(torch.bincount(self.data.hyperedge_index[1])) / self.data.num_edges
    def num_nodes(self):
        return self.data.num_nodes

    def num_edges(self):
        return self.data.num_edges

    def average_degree(self):
        return 2 * self.num_edges() / self.num_nodes()

    def clustering_coefficient(self):
        # compute clustering coefficient with networkx
        return nx.average_clustering(self.graph)
    def density(self):
        # calculate density of graph
        n = self.num_nodes()
        m = self.num_edges()
        return 2*m / (n*(n-1))

    def diameter(self):
        if nx.is_connected(self.graph):
            diameter = nx.diameter(self.graph)
        else:
            diameter = None
        return diameter
