import ensembl_rest
import igraph
import logging
import networkx as nx
from netZooPy.panda.panda import Panda
import numpy as np
import os
import pandas as pd
import re
import torch
import urllib.request
import numbers
from genome_graph import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrandGraph(object):
    """_summary_

    Args:
        network_link: the url used to download the tissue or diseases
    Usage:
        GrandGraph(network_link="")
    Returns:
        _description_
    """
    _ppi_link = 'https://granddb.s3.amazonaws.com/tissues/ppi/tissues_ppi.txt'
    _motif_link = 'https://granddb.s3.amazonaws.com/tissues/motif/tissues_motif.txt'
    _exp_link = 'https://granddb.s3.amazonaws.com/tissues/expression/Liver.csv'
    _network_link = 'https://granddb.s3.amazonaws.com/tissues/networks/Liver.csv'
    _tag_name = 'liver'

    max_ensembl_attempts = 5

    def __init__(
            self, tag_name=None,
            ppi_link=None, motif_link=None,
            exp_link=None, network_link=None, build=False):
        self.tag_name = tag_name
        self.ppi_link = ppi_link
        self.motif_link = motif_link
        self.exp_link = exp_link
        self.network_link = network_link
        if self.downloaded:
            logging.info(f"Data already downloaded for {self.tag_name}")
            path_name = self.get_network_name(self.path, self.network_link)
            network = pd.read_csv(path_name,index_col=0)
            logging.info(f"Loaded from {path_name}")
        else:
            logging.info(f"Downloading data for {self.tag_name}")
            file_name_map = self.download_links()
            logging.info(f"Downloaded data, storing in {self.path}")
            if not build:
                network = pd.read_csv(file_name_map['network'],index_col=0)
            else:
                logging.info('Building network, this might take a while.')
                panda_obj = Panda(
                    self.exp_link, self.motif_link, self.ppi_link)
                panda_obj.save_panda_results(self.path + '/panda_results.csv')
                network = pd.read_csv(self.path + '/panda_results.csv',index_col=0)
        # Standardise columns to Ensembl IDs if not already
        if not self.downloaded:
        #if not any([re.match('ENSG[0-9]+',el) for el in network.index]):
            logging.info("Converting network indices to Ensembl IDs...")
            good=False
            attempts=0
            while not good:
                try:
                    result = ensembl_rest.symbol_post(species='homo sapiens',
                                                 params={'symbols': list(network.index)})
                    good=True
                except Exception as e:
                    if attempts>=self.max_ensembl_attempts:
                        raise e
                    else:
                        logging.info("Ensembl rest failed: {}\nRetrying... ({}/{})".format(e,attempts,self.max_ensembl_attempts))
                        attempts+=1

            if not len(result.keys())==len(network.index):
                logging.info(f"Ensembl search returned {len(result)} out of {len(network.index)} "
                      f"queries. Missing {len(network.index)-len(result)}:\n"+
                      "\n".join(set(network.index).difference(result.keys())))
                # TODO: Implement other ways of identifying the Ensembl ID.
                logging.info("For now, we're just going to delete these. "
                      "TODO: Implement other ways of identifying the Ensembl ID.")
                network = network.loc[result.keys()]
                network.index = [result[el]['id'] for el in network.index]
                network.to_csv(self.path + '/panda_results.csv')
        self.network = pd.read_csv(self.path + '/panda_results.csv',header=0,index_col=0)
        logging.info(f'Loaded network with a shape of {network.shape}')

    def get_network_name(self, path, network_link):
        file_name = network_link.split('/')[-1]
        file_name = 'network_' + file_name
        return os.path.join(path, file_name)

    @property
    def path(self):
        path = os.path.join(DATA_DIR,'grand',self.tag_name)
        return path

    @property
    def tag_name(self):
        return self._tag_name

    @tag_name.setter
    def tag_name(self,tag_name):
        if tag_name is not None:
            self._tag_name = tag_name

    @property
    def ppi_link(self):
        return self._ppi_link

    @ppi_link.setter
    def ppi_link(self,ppi_link):
        if ppi_link is not None:
            self._ppi_link = ppi_link

    @property
    def motif_link(self):
        return self._motif_link

    @motif_link.setter
    def motif_link(self,motif_link):
        if motif_link is not None:
            self._motif_link = motif_link

    @property
    def exp_link(self):
        return self._exp_link

    @exp_link.setter
    def exp_link(self,exp_link):
        if exp_link is not None:
            self._exp_link = exp_link

    @property
    def network_link(self):
        return self._network_link

    @network_link.setter
    def network_link(self,network_link):
        if network_link is not None:
            self._network_link = network_link

    @property
    def links(self):
        return {
                #'ppi': self.ppi_link,
                #'motif': self.motif_link,
                #'exp': self.exp_link,
                'network': self.network_link
            }

    @property
    def downloaded(self):
        files = []
        for name, link in self.links.items():
            file_name = link.split('/')[-1]
            if name == 'network':
                file_name = 'network_' + file_name
            files.append(os.path.join(self.path, file_name))
        return all([os.path.exists(file) for file in files])


    @property
    def edge_idx(self):
        """ Constructs the non-symmetric adjacency matrix for the network.

        A_ij is nonzero if TF j is a regulator of gene i.
        """
        gene_set=self.nodes
        # This is slow, so we use searchsorted instead
        """
        index_indices=np.argwhere(np.asarray(self.network.index)[:,None]==np.asarray(gene_set)[None,:])[:,1]
        col_indices=np.argwhere(np.asarray(self.network.columns)[:,None]==np.asarray(gene_set)[None,:])[:,1]
        """
        index_indices=np.searchsorted(gene_set,np.sort(self.network.index))
        col_indices=np.searchsorted(gene_set,np.sort(self.network.columns))
        # Edge index is [2, num_edges], where [0,:] are source indices and
        # [1,:] are target indices
        sources = np.repeat(index_indices,len(col_indices))
        targets = np.tile(col_indices,len(index_indices))
        edges = np.stack([sources,targets],axis=-1)
        return edges

    @property
    def edges(self):
        edges = self.edge_idx
        edges = np.stack([self.nodes[edges[:,0]],self.nodes[edges[:,1]]],axis=-1)
        return edges

    @property
    def edge_weights(self):
        return np.asarray(self.network,dtype=np.float32).flatten()


    def weighted_edges(self,threshold=None):
        edges = self.edges
        weights = self.edge_weights
        if threshold is not None:
            idx = np.abs(weights)>=threshold
            edges = edges[idx]
            weights = weights[idx]

        return edges,weights

    def weighted_edge_idx(self,threshold=None):
        edges = self.edge_idx
        weights = self.edge_weights
        if threshold is not None:
            idx = np.abs(weights)>=threshold
            edges = edges[idx]
            weights = weights[idx]

        return edges,weights

    @property
    def nodes(self):
        return np.unique(list(self.network.index)+list(self.network.columns))

    @property
    def node_idx(self):
        return np.asarray(range(len(self.nodes)),dtype=np.int64)

    def to_networkx(self, abs_threshold=1.):
        raise NotImplementedError("Networkx crashes for even a small (100k) number of edges.")
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_weighted_edges_from([(*edge,weight) for edge, weight in zip(*self.weighted_edges(abs_threshold))])
        return g

    def to_igraph(self,abs_threshold=1.):
        edges,weights = self.weighted_edge_idx(abs_threshold)
        return igraph.Graph(n=len(self.nodes),edges=edges,
            edge_attrs={'weight':weights},directed=False)

    def download_links(self):
        files = {}
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        for name, link in self.links.items():
            file_name = link.split('/')[-1]
            if name == 'network':
                file_name = 'network_' + file_name
            save_file = os.path.join(self.path, file_name)
            if not os.path.exists(save_file):
                urllib.request.urlretrieve(link, save_file)

                # cmd = f'wget -p {path} {link}'
                # logging.info(cmd)
                # os.system(cmd)
                logging.info('Downloaded ' + link + 'to' + save_file)
                files[name] = save_file
        return files

def get_available_networks(family):
    assert family in ['drugs','cancers','tissues','cell']
    return pd.read_html(f"https://grand.networkmedicine.org/{family}/")[0]


def get_dataset_ensembl_ids(ensembl_file="", bed_annot_dir="") -> pd.DataFrame:
    """Get Ensembl IDs for sequences in a Basenji dataset.

    Generates files for the sequences containing the genome annotations, if they don't already exist.
    Args:
        ensembl_file (str): Organism. One of ['human', 'mouse]
        bed_annot_dir (str): GENCODE genome version. Defaults to hg38 for human, mm10 for mouse.
    Returns:
        pd.DataFrame: DataFrame

    Usage:
        bed_with_ensembl = genome_graph.data.basenji.get_dataset_ensembl_ids('human',41,num_procs=3,chunksize=100)
    """
    with open(ensembl_file, 'r') as f:
        # Read the lines
        cols = [len(l.split(',')) for l in f.readlines()]
        max_cols = max(cols)

    column_names = list(range(0, max_cols+1))

    ids = pd.read_csv(ensembl_file, header=None, names=column_names)

    ensembl_cols = [col for col in ids.columns if isinstance(col, numbers.Number)]

    ensembl_ids = []
    for i, row in ids[ensembl_cols].iterrows():
        # Matches alphanumeric up until a \.[0-9]
        ensembl_ids.append([re.search("[\w0-9]+(?=\.[0-9]+)",el).group(0) for el in row.dropna() if re.search("[\w0-9]+(?=\.[0-9]+)",el)])
    return ensembl_ids
