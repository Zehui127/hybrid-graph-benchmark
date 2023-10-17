from hg.hybrid_graph.io import get_dataset_single
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
# name = ['Brain', 'Lung', 'Breast', 'Artery_Coronary', 'Artery_Aorta', 'Stomach']
# dataset = Grand('/Users/lizehui/Desktop/workspace/neuraIPS rebuttle/hybrid_graph_benchmark/data/grand', name='Brain')
# dataset = Amazon('/Users/lizehui/Desktop/workspace/neuraIPS rebuttle/hybrid_graph_benchmark/data/amazon', name='Computers')
# dataset = Twitch('/Users/lizehui/Desktop/workspace/neuraIPS rebuttle/hybrid_graph_benchmark/data/musae', name='EN')


def jaccard_similarity(edge_index, hyperedge_index):
    """Compute Jaccard-based Hyperedge Overlap Score (J-HOS) for all hyperedges."""

    def jaccard_similarity(set1, set2):
        """Compute Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def hyperedge_to_nodes(hyperedge_id):
        """Get nodes of a specific hyperedge."""
        return set(hyperedge_index[0][hyperedge_index[1] == hyperedge_id].tolist())

    def simple_edges_of_nodes(node_set):
        """Get nodes connected by simple edges which are part of the node set."""
        edges = set(tuple(edge) for edge in edge_index.t().tolist())
        return {edge[0] for edge in edges if edge[0] in node_set or edge[1] in node_set}.union(
               {edge[1] for edge in edges if edge[0] in node_set or edge[1] in node_set})

    unique_hyperedges = torch.unique(hyperedge_index[1])
    scores = torch.zeros(unique_hyperedges.size(0))

    for idx in tqdm(range(len(unique_hyperedges))):
        hyperedge_id = unique_hyperedges[idx]
        hyperedge_nodes = hyperedge_to_nodes(hyperedge_id)
        simple_edge_nodes = simple_edges_of_nodes(hyperedge_nodes)
        scores[idx] = jaccard_similarity(hyperedge_nodes, simple_edge_nodes)
    return scores

def hyperedge_intersection_distribution(hyperedge_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyperedge_index = hyperedge_index.to(device)

    num_hyperedges = hyperedge_index[1].max().item() + 1
    num_vertices = hyperedge_index[0].max().item() + 1

    # Create a binary matrix representing hyperedges (on GPU if available)
    hyperedge_matrix = torch.zeros(num_hyperedges, num_vertices, dtype=torch.float, device=device)

    for vertex, edge_id in tqdm(hyperedge_index.T, desc='Processing Hyperedges', leave=False):
        hyperedge_matrix[edge_id, vertex] = 1.0  # Note the floating-point value here

    intersection_sizes = torch.mm(hyperedge_matrix, hyperedge_matrix.T)
    intersection_sizes.fill_diagonal_(0)

    intersection_sizes = intersection_sizes.int()
    distribution = torch.bincount(intersection_sizes.view(-1))
    distribution = distribution//num_hyperedges
    return distribution.cpu()

import torch
from concurrent.futures import ThreadPoolExecutor

def compute_j_hos_parallel(edge_index, hyperedge_index):

    def jaccard_similarity(set1, set2):
        """Compute Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def hyperedge_to_nodes(hyperedge_id, hyperedge_node_dict):
        """Get nodes of a specific hyperedge using precomputed dictionary."""
        return hyperedge_node_dict[hyperedge_id]

    def simple_edges_of_nodes(node_set, edges):
        """Get nodes connected by simple edges which are part of the node set."""
        return {edge[0] for edge in edges if edge[0] in node_set or edge[1] in node_set}.union(
               {edge[1] for edge in edges if edge[0] in node_set or edge[1] in node_set})

    # Construct edges set once
    edges = set(tuple(edge) for edge in edge_index.t().tolist())

    # Precompute hyperedge to nodes dictionary
    unique_hyperedges = torch.unique(hyperedge_index[1]).tolist()
    hyperedge_node_dict = {hyperedge: set(hyperedge_index[0][hyperedge_index[1] == hyperedge].tolist())
                           for hyperedge in unique_hyperedges}

    scores = torch.zeros(len(unique_hyperedges))

    # Define worker function for multithreading
    def worker(hyperedge_id):
        hyperedge_nodes = hyperedge_to_nodes(hyperedge_id, hyperedge_node_dict)
        simple_edge_nodes = simple_edges_of_nodes(hyperedge_nodes, edges)
        return jaccard_similarity(hyperedge_nodes, simple_edge_nodes)

    # Use ThreadPoolExecutor for parallel computation
    futures = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        print(f"Number of workers used: {executor._max_workers}")
        for hyperedge_id in unique_hyperedges:
            futures.append(executor.submit(worker, hyperedge_id))

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    for idx, score in enumerate(results):
        scores[idx] = score

    return scores


def plot_jaccard_similarity(stats, dataset_name):
    torch.save(stats, f'jaccard/{dataset_name}_jaccard.pt')
    print(torch.mean(stats))
    # plot the histogram
    # plot the histogram at log scale
    plt.hist(stats, bins=100, log=True)
    #set title and labels
    plt.title(f'Jaccard similarity for {dataset_name}')
    plt.xlabel('Jaccard similarity')
    plt.ylabel('Number of hyperedges')
    # add mean value to the plot
    plt.axvline(torch.mean(stats).item(), color='k',
                linestyle='dashed', linewidth=1)
    # label the mean value around the line horizontally
    plt.text(torch.mean(stats).item(), 1,
             f'mean={torch.mean(stats).item():.6f}',
             va='bottom', ha='center')
    plt.savefig(f'jaccard/{dataset_name}_jaccard.png')
    plt.close()

Datasets = ["musae_Twitch_ES","musae_Twitch_FR","musae_Twitch_EN",
            "musae_Twitch_PT","musae_Twitch_RU","musae_Twitch_DE",
            "grand_ArteryAorta","grand_ArteryCoronary","grand_Breast","grand_Brain",
            "grand_Leukemia","grand_Lung","grand_Stomach","grand_Lungcancer","grand_Stomachcancer",
            "grand_KidneyCancer","amazon_Photo","amazon_Computer",
            "musae_Wiki_squirrel","musae_Wiki_crocodile","musae_Wiki_chameleon",
            "musae_Facebook","musae_Github"]
# Datasets  = ["musae_Twitch_ES","musae_Twitch_FR","musae_Twitch_EN",
#             "musae_Twitch_PT","musae_Twitch_RU","musae_Twitch_DE",
#             "grand_ArteryAorta","grand_ArteryCoronary","grand_Breast","grand_Brain"]
# start from grand_Leukemia
# Datasets = ["grand_Leukemia","grand_Lung","grand_Stomach","grand_Lungcancer","grand_Stomachcancer",
#             "grand_KidneyCancer",/"amazon_Photo","amazon_Computer",
#             "musae_Wiki_squirrel","musae_Wiki_crocodile","musae_Wiki_chameleon"]
Datasets = ["musae_Wiki_chameleon","grand_Breast","amazon_Computer"]

for dataset in Datasets:
    data = get_dataset_single(dataset)
    print(data)
    distribution = hyperedge_intersection_distribution(data.hyperedge_index)
    print(distribution)
    # res = compute_j_hos_parallel(data.edge_index, data.hyperedge_index)
    # plot_jaccard_similarity(res, dataset)
    # print(res)
    # print(res.mean())

    # Prepare the data for plotting
    sizes = torch.arange(len(distribution))
    counts = distribution

    # Plot the distribution
    plt.bar(sizes, counts,)
    # plt.yscale('log')
    plt.xlabel('Intersection Size')
    plt.ylabel('Frequency (log scale)')
    plt.title('Hyperedge-Intersection-Size Distribution')
    plt.savefig(f'jaccard/new_{dataset}_overlap_dist.png')
    plt.close()
