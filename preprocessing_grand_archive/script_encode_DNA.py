from grand import GrandGraph, get_path, OneHotEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm

# Aim: create two complete csv files with the following columns:
# The first:
# original name, ESEB ID, Raw DNA sequence, ...
# The second:
# original name, ESEB ID, DNA sequence embedding, ...


def create_original_emb():
    df_emb = pd.read_csv("node_emb.csv", sep="\t")
    df_emb_curate = pd.read_csv("curate_complete_node_emb.csv", sep="\t")
    df_emb_curate['rid'] = df_emb['id'].values
    df_emb_curate = df_emb_curate.loc[:, ~df_emb_curate.columns.str.contains('^Unnamed')]
    df_emb_curate.to_csv("original_node_emb.csv", sep="\t")
def create_kmer_emb():
    df_emb = pd.read_csv("original_node_emb.csv", sep="\t")
    max_length = 0
    for seq in df_emb['seq']:
        if (not pd.isnull(seq)) and len(seq) > max_length:
            max_length = len(seq)
    kmer_emb = []
    encoder = OneHotEncoder(max_length=max_length)
    for seq in tqdm(df_emb['seq']):
        if pd.isnull(seq):
            kmer_emb.append("")
        else:
            v = encoder.kmer_encoder(seq)
            kmer_emb.append(v/np.linalg.norm(v))
    df_emb = df_emb.loc[:, ~df_emb.columns.str.contains('seq')]
    df_emb.to_csv("reference_node_emb.csv", sep="\t")

    df = pd.DataFrame(kmer_emb)
    df.to_csv("4mer_scaled_node_emb.csv", sep="\t")
def main():
    #iterate through all the graphs and delete all the row and columns containing the unseen nodes


if __name__ == "__main__":
    main()
