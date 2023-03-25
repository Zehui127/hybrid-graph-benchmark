import json
from grand import GrandGraph, get_path

json_data = json.load(open(get_path()+"/grand_config.json"))
for tissue, network_link in zip(json_data["cancername"], json_data["cancers"]):
    network = GrandGraph(tissue, network_link)
