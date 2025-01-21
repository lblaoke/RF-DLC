import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from collections import Counter

from medmnist import DermaMNIST
    
class DermaMNIST_LT_loader(DataLoader):
    def __init__(self, batch_size:int, num_workers:int=0):
        normalize = transforms.Normalize(mean=[0.7636, 0.5387, 0.5621], std=[0.1351, 0.1526, 0.1675])
        train_trsfm = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])

        self.dataset = DermaMNIST(split="train", download=True, transform=train_trsfm)
        self.val_dataset = DermaMNIST(split="val", download=True, transform=test_trsfm)

        num_classes = np.max(self.dataset.labels) + 1
        assert num_classes == 7

        label_counts = Counter(self.dataset.labels.reshape(-1).tolist())
        sorted_labels = [label for label, count in label_counts.most_common()]
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        self.dataset.labels = np.array([label_mapping[label] for label in self.dataset.labels.reshape(-1).tolist()], dtype=np.int64)
        self.val_dataset.labels = np.array([label_mapping[label] for label in self.val_dataset.labels.reshape(-1).tolist()], dtype=np.int64)

        self.dataset.targets = self.dataset.labels
        self.val_dataset.targets = self.val_dataset.labels

        self.cls_num_list = np.histogram(self.dataset.labels, bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=None)

    def split_validation(self):
        return DataLoader(
            dataset     = self.val_dataset,
            batch_size  = 4096,
            shuffle     = False,
            num_workers = 0,
            drop_last   = False
        )

if __name__ == '__main__':
    data = DermaMNIST(split="train", download=True)
    val_data = DermaMNIST(split="val", download=True)

    t1 = transforms.ToTensor()
    t2 = transforms.Resize(32)
    images = []

    for (img, _) in data:
        images.append(t1(t2(img)))
    for (img, _) in val_data:
        images.append(t1(t2(img)))

    images = torch.stack(images)

    # calculate mean over each channel (r,g,b)
    mean_r = images[:, 0, :, :].mean()
    mean_g = images[:, 1, :, :].mean()
    mean_b = images[:, 2, :, :].mean()
    print(mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = images[:, 0, :, :].std()
    std_g = images[:, 1, :, :].std()
    std_b = images[:, 2, :, :].std()
    print(std_r,std_g,std_b)
