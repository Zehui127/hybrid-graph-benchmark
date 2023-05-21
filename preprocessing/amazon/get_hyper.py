import os
import sys
import torch
from torch_geometric.data import download_url
from datasets.standard import GraphFormatter

url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id={}"

file_id = {
    'Computers': '1f388hEIrqbfKlo9VAJo9Rvixfi3ZUgZz',
    'Photos': '1LMmK1sLoOkXVb3c66I3bJJDsCoc1zoqD',
}

def download_original(name, path):
    if not os.path.exists(path):
        os.mkdir(path)
    data_file = download_url(url.format(file_id[name]), path, filename=name+'.pt')
    data = torch.load(data_file)
    return data, os.path.join(path, name+'.pt')

def process_and_save(data, path):
    hyper_data = GraphFormatter()(data)
    new_path = path.replace('.pt', '_hyper.pt')
    print('save hyper dataset in', new_path)
    torch.save(hyper_data, new_path)

if __name__ == '__main__':
    data, path = download_original('Photos', 'data/raw')
    process_and_save(data, path)
    data, path = download_original('Computers', 'data/raw')
    process_and_save(data, path)

