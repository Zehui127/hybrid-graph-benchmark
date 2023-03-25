import ensembl_rest
from grand import GrandGraph, get_path
import requests
import json
from tqdm import tqdm
import pandas as pd
import torch
from bs4 import BeautifulSoup
import re
from lxml import etree
def get_id(gene_symbol):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/human/{gene_symbol}?object_type=gene"
    ensembl_id = ''
    response = requests.get(server+ext, headers={"Content-Type": "application/json"})
    if response.ok:
        data = response.json()
        if len(data) > 0:
            ensembl_id = data[0]["id"]
            #print(f"The Ensembl ID for gene symbol {gene_symbol} is {ensembl_id}")
        else:
            print(f"No Ensembl ID found for gene symbol {gene_symbol}")
    else:
        response.raise_for_status()
    return ensembl_id
def get_missing_info(gene_symbol):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/id/human/{gene_symbol}"
    ensembl_id = ''
    response = requests.get(server+ext, headers={"Content-Type": "application/json"})
    if response.ok:
        data = response.json()
        if len(data) > 0:
            ensembl_id = data[0]["id"]
            #print(f"The Ensembl ID for gene symbol {gene_symbol} is {ensembl_id}")
        else:
            print(f"No Ensembl ID found for gene symbol {gene_symbol}")
    else:
        response.raise_for_status()
    return ensembl_id
def get_gene_info_bs(gene_id):
    url = f"https://feb2014.archive.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g={gene_id}"
    # Send a GET request to the API and get the response
    location = None
    biotype  = None
    sequence = None
    try:
        response = requests.get(url, headers={'Content-Type': 'application/json'})
        # Parse the JSON response
        # data = json.loads(response.text)
        data = response.text
        location,biotype,location_text = useBeautifulSoup(data)
        sequence = getSequenceFromLocation(location_text)
        if "Protein codingGenes" in biotype:
            biotype = "protein_coding"
    except:
         location = ""
         biotype = ""
         sequence = ""
    return location, biotype, sequence
def useBeautifulSoup(html_code):
    # assuming the HTML code you provided is stored in a variable called html_code
    soup = BeautifulSoup(html_code, 'html.parser')

    # find the div with class "summary_panel"
    summary_panel = soup.find('div', class_='summary_panel')

    # find the div with class "twocol"
    twocol = summary_panel.find('div', class_='twocol')

    # find the row that contains the location information
    location_element = twocol.find('div', {'class': 'lhs'}, string='Location')
    location_element_format = twocol.find('div', {'class': 'lhs'}, string='INSDC coordinates')
    # extract the location information
    location_text = location_element.find_next_sibling('div').text.strip()
    location_text_format = location_element_format.find_next_sibling('div').text.strip()
    # find the row that contains the biotype information
    table = soup.find('table', {'id': 'transcripts_table'})
    biotype = table.select_one('#transcripts_table tbody td:nth-child(6)').text.strip()
    return location_text_format, biotype, location_text

def getSequenceFromLocation(location):
    def getSequence2(chromosome, start, end):
        base = 'http://genome.ucsc.edu/cgi-bin/das/hg19/dna?segment='
        url = base + chromosome + ':' + str(start) + ',' + str(end)
        doc = etree.parse(url,parser=etree.XMLParser())
        if doc != '':
            sequence = doc.xpath('SEQUENCE/DNA/text()')[0].replace('\n','')
        else:
            sequence = 'THE SEQUENCE DOES NOT EXIST FOR GIVEN COORDINATES'
        return sequence
    def parseLocation(location):
        string_without_comma = location.replace(",", "")
        pattern = r"\d+"
        matches = re.findall(pattern, string_without_comma)
        return matches
    matches = parseLocation(location)
    return getSequence2(matches[0],matches[1],matches[2]).upper()
def updateMissing():
    df = pd.read_csv("node_emb.csv", sep='\t')
    temp_dict = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    pbar = tqdm(total=45011)
    for index, row in tqdm(df.iterrows()):
        if not row['id'].startswith('ENSG'):
            missing_id = get_id(row['id'])
            if len(missing_id) > 2:
                temp_dict['id'] = missing_id
                gene_seq_info = get_gene_sequence(missing_id)
                temp_dict['biotype'] = get_gene_info(missing_id)
                temp_dict['desc'] = gene_seq_info['desc']
                temp_dict['seq'] = gene_seq_info['seq']
                temp_dict['molecule'] = gene_seq_info['molecule']
                df.loc[index] = [index,temp_dict['id'],temp_dict['desc'],temp_dict['seq'],temp_dict['biotype'],temp_dict['molecule'] ]
        pbar.update(1)
    df.to_csv('complete_node_emb.csv', sep='\t')

def updateMissing2():
    df = pd.read_csv("complete_node_emb.csv", sep='\t')
    temp_dict = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    pbar = tqdm(total=45011)
    for index, row in tqdm(df.iterrows()):
        if row['biotype'] == "Undefined":
            location,biotype,sequence = get_gene_info_bs(row['id'])
            if len(location) > 2:
                temp_dict['id'] = row['id']
                temp_dict['biotype'] = biotype
                temp_dict['desc'] = location
                temp_dict['seq'] = sequence
                temp_dict['molecule'] = "Currated"
                df.loc[index] = [index,index,temp_dict['id'],temp_dict['desc'],temp_dict['seq'],temp_dict['biotype'],temp_dict['molecule'] ]
        pbar.update(1)
    df.to_csv('curate_complete_node_emb.csv', sep='\t')

def get_gene_info(gene_id):
    gene_type = "Undefined"
    try:
        # Set the species and gene ID
        species = 'homo_sapiens'
        # Set the REST API URL with the species and gene ID
        url = f'http://rest.ensembl.org/lookup/id/{gene_id}?expand=1;content-type=application/json;species={species}'
        # Send a GET request to the API and get the response
        response = requests.get(url, headers={'Content-Type': 'application/json'})
        # Parse the JSON response
        data = json.loads(response.text)
        # Get the gene type from the response

        if 'biotype' in data:
            gene_type = data['biotype']
        # Print the gene type
    except:
        gene_type = "Undefined"
    return gene_type


def get_gene_sequence(gene_id):
    data = None
    try:
        data = ensembl_rest.sequence_id(gene_id)
    except:
        data = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    return data


def process_graph(nodes, dest="node_emb.csv"):
    temp_dict = {'id': "", 'desc': "", 'seq': "", 'biotype': "", "molecule": ""}
    df = pd.DataFrame(columns=['id', 'desc', 'seq', 'biotype','molecule'], dtype=str)
    count = 0
    for node in tqdm(nodes):
        # get gene position and seq
        temp_dict['biotype'] = get_gene_info(node)
        # get the rest info
        gene_seq_info = get_gene_sequence(node)
        temp_dict['id'] = node#gene_seq_info['id']
        temp_dict['desc'] = gene_seq_info['desc']
        temp_dict['seq'] = gene_seq_info['seq']
        temp_dict['molecule'] = gene_seq_info['molecule']
        df.loc[count] = [temp_dict['id'],temp_dict['desc'],temp_dict['seq'],temp_dict['biotype'],temp_dict['molecule'] ]
        count += 1
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
    updateMissing2()
    #nodes = torch.load("genes.pt")
    #process_graph(nodes)
    #df = pd.read_csv("node_emb.csv",sep="\t")
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
