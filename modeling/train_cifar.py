from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import numpy as np
from datetime import datetime
import torch
from PIL import Image
from simple_net import SimpleNet
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

channel_means = (0.4914, 0.4822, 0.4465)
channel_stds = (0.2023, 0.1994, 0.2010)
# suggested transformations:
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    #transforms.Normalize(channel_means, channel_stds),
])

transform_test = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Normalize(channel_means, channel_stds),
])

# Normalize is applying (value - mean) / std for every pixel in every channel.
# The means and deviations for every channel of cifar are provided above.

# alternate dataset: these come in the format of HWC, when pulling in
# data from disk it needs to be reshaped into that format.

class CifarDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.paths = [sample['image'] for sample in samples]
        self.labels = [sample['label_num'] for sample in samples]
        self.base = "/home/pdoyle/workspace/neural_nets/simple_net_cifar/" # work on this.
        self.transform = transform
        self.channel_means = (0.4914, 0.4822, 0.4465)
        self.channel_stds = (0.2023, 0.1994, 0.2010)
        self.norm = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(self.channel_means, self.channel_stds)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw_img = Image.open(self.base + self.paths[idx])
        if self.transform:
            raw_img = self.transform(raw_img)
        img = np.asarray(raw_img) / 255
        img = img.astype(np.float)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        #img = torch.transpose(img, 0, 2, 3, 1)

        labels = np.array(self.labels[idx]).astype(np.double)
        return img, labels

def _get_dataset_loader(data, transform=None, shuffle=False):
    return torch.utils.data.DataLoader(
            CifarDataset(data, transform=transform), batch_size=64, shuffle=shuffle, num_workers=8)

def main():
    es_staged_data_index = "cifar-metadata-1"
    es_logging_index = "custom-net-cifar-12"
    output_model = es_logging_index + ".pth"
    es = Elasticsearch("localhost:9200")
    data = [doc["_source"] for 
                doc in list(
                    scan(es, index=es_staged_data_index))]

    np.random.seed(42)
    np.random.shuffle(data)
    training_data = [x for x in data if "train" in x["set_type"]]
    testing_data = [x for x in data if "test" in x["set_type"]]
    print(f"Size of training set: {len(training_data)}")
    print(f"Size of testing set: {len(testing_data)}")

    # didnt use this time around.
    train_dataset_loader = _get_dataset_loader(training_data, transform=transform_train, shuffle=True)
    test_dataset_loader = _get_dataset_loader(testing_data)

    net = SimpleNet(10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    # Train
    print("training...")
    for epoch in range(300):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            outputs = net(inputs.float().cuda())
            loss = criterion(outputs, labels.long().cuda())
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            print_on = 100
            if (i + 1) % print_on == 0:
                record = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "cross-entropy-loss": running_loss/print_on,
                        "model-name": "train-simplenet-8"
                        }
                es.index(index=es_logging_index, body=record)
                print('[%d, %5d] loss %.3f' % (epoch + 1, i + 1, running_loss / (print_on + 1)))
                running_loss = 0.0

        # Test
        if epoch + 1 % 10:
            print("testing...")
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                i = 0.0
                for inputs, labels in test_dataset_loader:
                    outputs = net(inputs.float().cuda())
                    #_, predicted = torch.max(outputs.data, 1)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.cuda()).sum().item()
                    #correct += (predicted == labels.long().cuda()).sum().item()
                    i += 1

                test_accuracy = correct / total
            print(f"Test Accuracy: {test_accuracy}")
            print(f"Correct: {correct}, Incorrect: {total-correct}")
            record = {
                    "accuracy": test_accuracy,
                    "correct": correct,
                    "incorrect": total-correct,
                    "timestamp": datetime.utcnow().isoformat()}
            es.index(index=es_logging_index, body=record)
    
    # Save
    print("saving...")
    torch.save(net.state_dict(), output_model)


if __name__ == '__main__':
    main()

