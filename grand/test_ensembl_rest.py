import ensembl_rest
from grand import GrandGraph, get_path
import requests
import json
from tqdm import tqdm
import pandas as pd
import torch


def get_gene_info(gene_id):

    # Set the species and gene ID
    species = 'homo_sapiens'
    # Set the REST API URL with the species and gene ID
    url = f'http://rest.ensembl.org/lookup/id/{gene_id}?expand=1;content-type=application/json;species={species}'
    # Send a GET request to the API and get the response
    response = requests.get(url, headers={'Content-Type': 'application/json'})
    # Parse the JSON response
    data = json.loads(response.text)
    # Get the gene type from the response
    gene_type = "Undefined"
    if 'biotype' in data:
        gene_type = data['biotype']
    # Print the gene type
    return gene_type


def get_gene_sequence(gene_id):
    data = None
    try:
        data = ensembl_rest.sequence_id(gene_id)
    except ensembl_rest.HTTPError as err:
        data = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    return data


def process_graph(nodes, dest="node_emb.pt"):
    temp_dict = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    for node in tqdm(nodes):
        # get gene position and seq
        temp_dict['biotype'] = get_gene_info(node)
        # get the rest info
        gene_seq_info = get_gene_sequence(node)
        temp_dict['id'] = gene_seq_info['id']
        temp_dict['desc'] = gene_seq_info['desc']
        temp_dict['seq'] = gene_seq_info['seq']
        temp_dict['molecule'] = gene_seq_info['molecule']

    df = pd.DataFrame.from_dict(temp_dict, orient='index',
                                columns=['id', 'desc', 'seq', 'biotype'])
    df.to_csv(dest, sep='\t')


def main():
    json_data = json.load(open(get_path()+"/grand_config.json"))
    iter_list = [["cancername", "cancers"], ["tissuename", "tissues"]]
    nodes = set()
    for ele in iter_list:
        for tissue, network_link in zip(json_data[ele[0]], json_data[ele[1]]):
            print(tissue, network_link)
            network = GrandGraph(tissue, network_link)
            # add all nodes into set
            for node in network.nodes:
                nodes.add(node)
    torch.save(nodes,'genes.pt')
    # process the nodes one by one
    process_graph(nodes)


if __name__ == "__main__":
    nodes = torch.load("genes.pt")
    process_graph(nodes)
#network = GrandGraph("Adipose_Subcutaneous", "https://granddb.s3.amazonaws.com/tissues/networks/Adipose_Subcutaneous.csv")
#process_graph(network.nodes)








"""
def count_type(nodes):
    def count_dict(content, type):
        if type in content:
            content[type] += 1
        else:
            content[type] = 1
    content = {}
    count = 10000
    for node in tqdm(nodes):
        count_dict(content, get_gene_info(node))
        count -= 1
        if count <= 0:
            break
    return content
"""
