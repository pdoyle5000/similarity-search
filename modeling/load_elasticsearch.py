# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
# The above url shows the few commands to pull down and run docker-elasticsearch

import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

es = Elasticsearch("localhost")

def _get_set_type(path):
    if "test_batch" in path:
        return "test"
    return "train"

def build_doc(record):
    es_index = "cifar-metadata-1"
    return {
        "_index": es_index,
        "_source": {
            "es_index": es_index,
            "image": record["image"],
            "set_type": _get_set_type(record["image"]),
            "label_text": record["label_text"],
            "label_num": record["label"],
            "timestamp": record["timestamp"]}}


with open('cifar_data.json') as json_file:  
    cifar_data = json.load(json_file)
    resp = list(parallel_bulk(es, [build_doc(record) for record in cifar_data]))
    print(resp)
