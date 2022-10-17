"""
Author: HuangFuSL
ID: 2022311931
Date: 2022-10-16

This is an implementation using torch. For the implementation without using torch.nn.Linear, see HW02-1-nn.py

NOTE: The implementation is NOT based on torch.nn, nor torch.optim. Actually, it does not use autograd feature. Usage of dataset and dataloader is limited to loading data in a corresponding manner.
"""
import csv
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
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
        self.y = torch.from_numpy(
            (self.data['Iris_type'] == 'Iris-setosa').astype(int).to_numpy(dtype=np.float32))
        self.X = torch.from_numpy(self.data[fields].to_numpy(dtype=np.float32))

    def __getitem__(self, index: int):
        return self.X[index], self.y[index: index + 1]

    def __len__(self):
        return self.X.shape[0]


DataLoader = torch.utils.data.dataloader.DataLoader


def sigmoid(x: torch.Tensor):
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x: torch.Tensor):
    return sigmoid(x) * (1 - sigmoid(x))


def loss(x: torch.Tensor, y: torch.Tensor):
    return -(y * torch.log(x) + (1 - y) * torch.log(1 - x))


def loss_derivative(x: torch.Tensor, y: torch.Tensor):
    return (1 - y) / (1 - x) - y / x


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    return x @ w + b


class Model():

    def __init__(self, *dim: int):
        self.w, self.b, self.i = [], [], []
        for x, y in zip(dim, dim[1:]):
            self.w.append(torch.randn(x, y))
            self.b.append(torch.randn(y))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Corresponds to torch.nn.Module implementation """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.i = []
        for w, b in zip(self.w, self.b):
            self.i.append(torch.clone(x))
            x = sigmoid(linear(x, w, b))
        return x

    def step(self, y_pred: torch.Tensor, y: torch.Tensor, lr: float):
        d = loss_derivative(y_pred, y)
        for w, b, x in zip(self.w[::-1], self.b[::-1], self.i[::-1]):
            d = sigmoid_derivative(linear(x, w, b)) * d
            w -= lr * (x.T @ d)
            b -= lr * (torch.ones((1, 7)) @ d).reshape(-1)
            d = d @ w.T
        return loss(y_pred, y).sum().item()


def train(model, data, test_data, lr, epochs):
    for epoch in range(epochs):
        l = 0
        for x, y in data:
            y_pred = model(x)
            acc = torch.mean((y_pred.round() == y).to(torch.float32))
            l += model.step(y_pred, y, lr)
        print(l)

        correct, union = 0, 0
        for x, y in test_data:
            y_pred = model(x)
            correct += torch.sum((y_pred.round() ==
                                  y).to(torch.float32)).item()
            union += y.shape[0]

        print(correct / union)


if __name__ == "__main__":
    train_data, test_data = map(IrisDataset, load_dataset(DATA_PATH))
    train_loader = DataLoader(train_data, 7, True)
    test_loader = DataLoader(test_data, 7, True)
    model = Model(4, 3, 1)
    train(model, train_loader, test_loader, 0.1, 7)
