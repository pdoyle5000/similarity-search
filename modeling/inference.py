import torch
from simple_net import SimpleNet

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import numpy as np
from datetime import datetime
from PIL import Image
from simple_net import SimpleNet
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from forward_hook import SaveFeatureVectors
from datetime import datetime
from datasketch import MinHash, MinHashLSH

class InferenceDataset(Dataset):
    def __init__(self, samples):
        self.paths = [sample['image'] for sample in samples]
        self.labels = [sample['label_num'] for sample in samples]
        self.base = "/home/pdoyle/workspace/neural_nets/simple_net_cifar/"
        self.c_means = (0.4914, 0.4822, 0.4465)
        self.c_stds = (0.2023, 0.1994, 0.2010)
        self.xform = transforms.Compose([
            transforms.Normalize(self.c_means, self.c_stds)])
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw_img = Image.open(self.base + self.paths[idx])
        image = np.asarray(raw_img) / 255
        img = image.astype(np.float)

        # re-order the dimensions, images open as HWC, change to CHW for the network input.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = self.xform(img)
        labels = np.array(self.labels[idx]).astype(np.int)
        return img, labels, self.paths[idx]

def _get_test_batch():
    es = Elasticsearch("localhost:9200")
    es_staged_data_index = "cifar-metadata-1"
    data = [doc["_source"] for
            doc in list(
                scan(es, index=es_staged_data_index))]
    testing_data = [x for x in data if "test" in x["set_type"]]
    return torch.utils.data.DataLoader(
            InferenceDataset(testing_data[0:100]), batch_size=100)

def _get_single_image_batch(img_path):
    single_img = {
        "image": img_path,
        "label_num": 0}
    return torch.utils.data.DataLoader(
        InferenceDataset([single_img]), batch_size=1)



def execute_inference(create_hashes=False, num_algs=32, hash_index="lsh-demo-3"):
    path = "../saved_models/cifar_model_300_epochs.pth"
    model = SimpleNet(10)
    model.load_state_dict(torch.load(path))
    
    # Init forward hook for exposing dense layer vectors
    sf = SaveFeatureVectors(model.conv13[2])
    model.eval()
    is_correct = 0.0
    path_list = []
    for inputs, labels, paths in _get_test_batch():
        outputs = model(inputs.float())
        _, pred = outputs.max(1)
        is_correct += (pred == labels.long()).sum().item()
        print(f"Num correct: {is_correct}")
        path_list.extend(paths)

    print("Extracting Vectors")
    if create_hashes:
        generate_hash_tables(path_list, sf.features)

def generate_hash_tables(path_list, features, num_algs=32):
    feature_mapping = [{"image": p, "vector": v} for p, v in zip(path_list, features)]
    #es = Elasticsearch("localhost")
    lsh = MinHashLSH(threshold=0.5, num_perm=num_algs)
    #print(f"Generating and Uploading Hashes to {hash_index}")
    for case in feature_mapping:
        m = MinHash(num_perm=num_algs)
        m.update(case["vector"].flatten())
        # TRY STUFF HEREEREREERER
        lsh.insert(case["image"], m)

        vector = {
            f"h{i+1}": h for i, h in enumerate(m.hashvalues.tolist())}
        vector["image"] = case["image"]
        #vector["timestamp"] = datetime.utcnow().isoformat()
        #doc_id = vector["image"].replace("/", "_", -1)
        #es.create(index=hash_index, body=vector, id=doc_id)
    #print("Hash Generation Complete.")
    #print(dir(lsh))
    #print(lsh.b)
    for a in lsh.hashtables[0]:
        print(f'{a.hex()}')



def execute_similarity_search(img_path, num_algs=32):
    # this part can be refactored into a "setup model" func
    # the func can have a flag for initing the hook or not.
    path = "../saved_models/cifar_model_300_epochs.pth"
    model = SimpleNet(10)
    model.load_state_dict(torch.load(path))
    # Init forward hook for exposing dense layer vectors
    sf = SaveFeatureVectors(model.conv13[2])
    model.eval()
    is_correct = 0.0
    path_list = []
    for inputs, labels, paths in _get_single_image_batch(img_path):
        outputs = model(inputs.float())
        _, pred = outputs.max(1)
        is_correct += (pred == labels.long()).sum().item()
        print(f"Num correct: {is_correct}")
        path_list.extend(paths)

    # Generate Hash: todo-try stuff here.
    m = MinHash(num_perm=num_algs)
    m.update(sf.features.flatten())
    #search_db(m.hashvalues.tolist())

def search_db(hashes, num_neighbors, hash_index="lsh-demo-3"):
    vector = {
        f"h{i+1}": h for i, h in enumerate(hashes)}

    # Generate Term Query
    query = {
            "size": num_neighbors,
            "query": {
                "bool": {
                    "should": _generate_terms(vector)}}}
    es = Elasticsearch("localhost")
    return es.search(index=hash_index, body=query)


def _generate_terms(vector_dict):
    return [{"term": {key: val}} for key, val in vector_dict.items()]

if __name__ == "__main__":
    execute_inference(True)
    #execute_similarity_search("cifar/data_batch_1/2351.png")
