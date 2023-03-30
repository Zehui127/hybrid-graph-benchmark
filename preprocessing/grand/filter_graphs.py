from grand import GrandGraph, get_path, OneHotEncoder, get_dataset_ensembl_ids
from torch_geometric.data import Data
from collections import OrderedDict
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor


def find_top_k_graph():
    """order the graph by number of edges
    Args:
        k: return the top k graph
    Returns:
        The list of selected graphs [(graph_name,path)]
    """
    target_graphs = OrderedDict()
    json_data = json.load(open(get_path()+"/grand_config.json"))
    iter_list = [["cancername", "cancers"], ["tissuename", "tissues"]]
    for ele in iter_list:
        # get graphs from cancer/tissue
        for tissue, network_link in zip(json_data[ele[0]], json_data[ele[1]]):
            network = GrandGraph(tissue, network_link)
            target_graphs[(tissue, network_link)] = len(network.edges)
    return target_graphs


def get_top_k(target_graphs):
    sorted_items = sorted(target_graphs.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(columns=['name', 'path', 'edgeNum'], dtype=str)
    count = 0
    for (name, path), i in sorted_items:
        df.loc[count] = [name, path, i]
        count += 1
    df.to_csv("graph_priority_list.csv", sep='\t')


def delete_undefined_nodes(graph, df_node):
    """delete the undefined nodes from graph.network
    Args:
        graph: the original graph
    Return:
        new_graph: the graph without the undefined nodes
    """
    new_network = graph.network
    for node in tqdm(graph.nodes):
        row_index = df_node.index[df_node['rid'] == node][0]
        row = df_node.iloc[row_index]
        if row['biotype'] == "Undefined":
            if node in new_network.columns:
                new_network = new_network.drop(columns=[node])
            if node in new_network.index:
                new_network = new_network.drop([node])
    graph.network = new_network
    return graph


def build_graph(graph, df_node, df_emb, hyper_edge_list, sparse_adj=True):
    """build the graph from edge index, and
       numpy data frame
    Args:
        edge_index: the edge_index of the actual graph

    Returns:
        x: the node embedding
        y: the node type; 1-> protein encoding, 0-> non-protein encoding
        adj: the adjacency matrix of the graph
    """
    print("Deleting undefined nodes...")
    graph = delete_undefined_nodes(graph, df_node)
    print("Finish Deleting; Convert graph into geometric Data...")
    y_mapping = {"protein_coding":1,"lncRNA":0}
    sample_emb = df_emb[0]
    nodes = graph.nodes    # map node id to node name
    node_index_dict = {node: i for i, node in enumerate(nodes)}
    x = torch.zeros((len(nodes), sample_emb.shape[0]), dtype=torch.float)
    y = torch.zeros(len(nodes), dtype=torch.long)
    for i, node in enumerate(nodes):
        # find in the df_node the row index which has a value of node on the column "rid"
        row_index = df_node.index[df_node['rid'] == node][0]
        row = df_node.iloc[row_index]

        emb = df_emb[row_index]
        bio_type = None
        if row['biotype'] in y_mapping:
            bio_type = y_mapping[row['biotype']]
        else:
            bio_type = 2
        x[i] = torch.Tensor(emb)
        y[i] = bio_type
    edges, weights = graph.weighted_edges(threshold=1.5)
    edge_index = torch.tensor([[node_index_dict[edge[0]], node_index_dict[edge[1]]] for edge in edges], dtype=torch.long).t().contiguous()
    hyperedge_index = add_hyper_edge(node_index_dict, hyper_edge_list)
    if sparse_adj:
        return  Data(x=x, y=y, adj=edge_index,
                     sparse_adj=SparseTensor.from_edge_index(edge_index, torch.Tensor(weights)),
                     hyperedge_index=hyperedge_index,
                     sparse_hyperedge_index=SparseTensor.from_edge_index(hyperedge_index))
    return Data(x=x, y=y, adj=edge_index, hyperedge_index=hyperedge_index)


def add_hyper_edge(node_index_dict, hyperedge_list):
    """ Add hyperedge_index to Data
    The format of hyper edge follows the following convention:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html#torch_geometric.nn.conv.HypergraphConv
    hyperedge_index = torch.tensor([
                                    [0, 1, 2, 1, 2, 3],
                                    [0, 0, 0, 1, 1, 1],
                                   ])
    Args:
        graph_data: original graph
    Returns:
        hyperedge
    """
    hyperedge_index = []
    for i, node_list in enumerate(hyperedge_list):
        added_hyper_edges = [[node_index_dict[id],i] for id in node_list if id in node_index_dict]
        hyperedge_index.extend(added_hyper_edges)
    return torch.tensor(hyperedge_index, dtype=torch.long).t().contiguous()


def main():
    #np.array(df['a'].tolist())
    #target_graphs = find_top_k_graph()
    df_node = pd.read_csv("reference_node_emb.csv", sep="\t", index_col=1)
    df_emb = pd.read_csv("4mer_scaled_node_emb.csv", sep="\t", index_col=0)
    emb = np.array(df_emb)
    df_graph_list = pd.read_csv("graph_priority_list.csv", sep="\t")
    hyper_edge_list = get_dataset_ensembl_ids()

    graphs_list = zip(df_graph_list["name"],df_graph_list["path"])
    for name, path in graphs_list:
        source_graph = GrandGraph(name, path)
        hyper_graph = build_graph(source_graph, df_node,emb, hyper_edge_list)
        torch.save(hyper_graph, f"grand_graph/{name}.pt")
    """
    for name, path in target_graphs:
        source_graph = GrandGraph(name,path)
        original_graph = build_graph(source_graph)
        hyper_graph = add_hyper_edge(original_graph)
        torch.save(hyper_graph,f"{name}.pt")
    """

if __name__ == "__main__":
    og = main()
