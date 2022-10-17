"""
Author: HuangFuSL
ID: 2022311931
Date: 2022-10-16

This is an implementation using torch.nn.Linear. For the implementation without using torch, see HW02-1.py
"""

import csv
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn
import torch.optim
import torch.utils.data.dataloader
import torch.utils.data.dataset

# Constants
POSITIVE_CATEGORY = 'Iris-setosa'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'iris.csv')
TEST_RATIO = 0.3

# Data loader
def load_dataset(path: str):
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    return sklearn.model_selection.train_test_split(data, test_size=TEST_RATIO)


class IrisDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, data: List[Dict[str, Any]]):
        fields = ['Sep_len', 'Sep_wid', 'Pet_len', 'Pet_wid']
        self.data = pd.DataFrame(data)
        self.y = torch.from_numpy((self.data['Iris_type'] == 'Iris-setosa').astype(int).to_numpy(dtype=np.float32))
        self.X = torch.from_numpy(self.data[fields].to_numpy(dtype=np.float32))

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

DataLoader = torch.utils.data.dataloader.DataLoader

class Model(torch.nn.Module):
    def __init__(self, *dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim[i], dim[i + 1])
            for i in range(len(dim) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x.reshape(-1)

def train(model, data, test_data, lr, epochs):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        _loss = 0
        for x, y in data:
            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            _loss += l.item()
            acc = torch.mean((torch.round(y_pred) == y).to(torch.float32))
            l.backward()
            optimizer.step()
        print(_loss)

        model.eval()
        correct, union = 0, 0
        for x, y in test_data:
            y_pred = model(x)
            correct += torch.sum((torch.round(y_pred) == y).to(torch.float32)).item()
            union += y.shape[0]
        print(correct / union)

if __name__ == "__main__":
    train_data, test_data = map(IrisDataset, load_dataset(DATA_PATH))
    train_loader = DataLoader(train_data, 7, True)
    test_loader = DataLoader(test_data, 7, True)
    model = Model(4, 3, 1)
    train(model, train_loader, test_loader, 0.1, 7)
