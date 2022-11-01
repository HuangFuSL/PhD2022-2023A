'''
Write python code to perform following tasks:

1. Construct three different MLP models for the dataset Titanic for classification
2. Define the structure of the neural networks
3. Compare their performance

Author: HuangFuSL
ID: 2022311931
Date: 2022/10/23
'''

from __future__ import annotations
import collections
import csv
import os
import sys
import functools
import typing as T
import typing_extensions as TE
from numbers import Number

import pandas as pd
import sklearn.model_selection
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data.dataloader

#! Change root directory
os.chdir(os.path.dirname(__file__))

#! Type Definitions and class aliases
T_OUTPUT = T.Optional[T.TextIO]
T_RECORD = T.Mapping[str, T.Any]
T_DATA = T.Sequence[T_RECORD]
T_TRANSLATION = T.Optional[T.Callable[[T.Any], T.Any]]
T_TRANS_TABLE = T.Mapping[str, T_TRANSLATION]
T_TENSOR = torch.Tensor
T_ACTIVATION = TE.Literal['relu', 'sigmoid', 'tanh']
C_Module = torch.nn.Module
C_Dataloader = torch.utils.data.dataloader.DataLoader

#! Constants and super parameters
DATA_PATH = r'titanic-3.csv'  # Depends on workspace configuration
DEFAULT_TRANS_TABLE: T_TRANS_TABLE = {
    'Passenger Class': lambda _: {'Third': 3, 'Second': 2, 'First': 1}[_],
    'Name': None,  # Skip
    'lastName': None,  # Skip
    'Sex': lambda _: {'Male': 1, 'Female': 0}[_],
    'Age': lambda _: float(_) if _ else -1,
    'No of Siblings or Spouses on Board': int,
    'No of Parents or Children on Board': int,
    'Ticket Number': None,  # Skip
    'Passenger Fare': lambda _: float(_) if _ else 0,
    'Cabin': lambda _: {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}[_[0]] if _ else 0,
    'Port of Embarkation': lambda _: {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2, '': -1}[_],
    'Life Boat': lambda _: int(bool(_)),
    'Survived': lambda _: {'No': 0, 'Yes': 1}[_],
}
DATALOADER_PARAMS = {
    'batch_size': 16,
    'shuffle': True,
}
TRAINING_PARAMS = {
    'lr': 0.01,
    'epochs': 10,
    'loss': torch.nn.BCELoss,
    'optim': torch.optim.SGD,
}

#! Data Functions


def translate_data(data: T_DATA, trans_table: T_TRANS_TABLE) -> T_DATA:
    ''' Perform preprocessing based on translation table '''
    ret = []
    for record in data:
        new_record = dict(record)
        for field, trans in trans_table.items():
            if field not in record:
                continue
            elif trans is None:
                new_record.pop(field)
            else:
                new_record[field] = trans(new_record[field])
        ret.append(new_record)
    return ret


def preprocess_data(data: T_DATA, trans_table: T.Optional[T_TRANS_TABLE] = None) -> T_DATA:
    if trans_table is not None:
        return translate_data(data, trans_table)
    return data  # directly return


def from_csv(path: str, test_ratio: float, trans_table: T.Optional[T_TRANS_TABLE] = None) -> T.Tuple[T_DATA, T_DATA]:
    ''' Load data from csv file '''
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = preprocess_data(list(reader), trans_table)

    train, test = sklearn.model_selection.train_test_split(
        data, test_size=test_ratio)
    return train, test


def preview_field(data: T_DATA, field: str, file: T_OUTPUT = sys.stdout):
    assert len(data) > 0
    records = [record[field] for record in data]
    if isinstance(records[0], (str, bytes, bytearray)):
        print(collections.Counter(records), file=file)
    elif isinstance(records[0], Number):
        print(pd.Series(records).describe(), file=file)


def preview_data(data: T_DATA, file: T_OUTPUT = sys.stdout):
    sep = '----------------------'
    print(sep, file=file)
    for field in data[0].keys():
        print(f'Field: {field}', file=file)
        preview_field(data, field, file)
        print(sep, file=file)


def xy_split(data: T_DATA, y_field: str) -> T.Tuple[T_TENSOR, T_TENSOR]:
    # Used as collate function in DataLoader
    x_df = pd.DataFrame(data)
    y_df = x_df.pop(y_field)
    x = torch.tensor(x_df.values, dtype=torch.float32)
    y = torch.tensor(y_df.values, dtype=torch.float32).reshape(-1, 1)
    return x, y

#! Model Construction


class Model(C_Module):

    _activation: T.Dict[str, T.Callable[[T_TENSOR], T_TENSOR]] = {
        'relu': torch.relu,
        'sigmoid': torch.sigmoid,
        'tanh': lambda _: torch.tanh(_) / 2 + 0.5,
    }

    def __init__(self, idim: int, hdim: T.Sequence[int], odim: int, activation: T_ACTIVATION):
        super().__init__()
        self.activation = self._activation[activation]
        l, r = [idim] + list(hdim), list(hdim) + [odim]
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(*params) for params in zip(l, r)
        )

    def forward(self, x: T_TENSOR) -> T_TENSOR:
        for _ in self.layers:
            x = self.activation(_(x))
        return x


Titanic_Model = functools.partial(Model, idim=9, odim=1)
# ^ Model1: 9 -> 16 -> 1, activation: tanh
Model_1 = functools.partial(Titanic_Model, hdim=[16], activation='tanh')
# ^ Model2: 9 -> 16 -> 1, activation: sigmoid
Model_2 = functools.partial(Titanic_Model, hdim=[16], activation='sigmoid')
# ^ Model3: 9 -> 6 -> 6 -> 1, activation: tanh
Model_3 = functools.partial(Titanic_Model, hdim=[6, 6], activation='tanh')

#! Training and testing


def train_model(model: C_Module, data: C_Dataloader, **params: T.Any) -> C_Module:
    model.train()
    optimizer = params['optim'](model.parameters(), lr=params['lr'])
    loss_fn = params['loss']()
    for _ in range(params['epochs']):
        for x, y in data:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
    return model


def test_model(model: C_Module, data: C_Dataloader) -> float:
    model.eval()
    ret = torch.tensor([], dtype=torch.float32)
    for x, y in data:
        y_pred = model(x)
        ret = torch.concat([ret, (y_pred.round() == y)])
    return ret.mean().item()


if __name__ == '__main__':
    sample, _ = from_csv(DATA_PATH, 0.2)
    with open('preview.txt', 'w') as f:
        '''
        Preview result:
        Passenger Class: Third, First, Second
        Name: String, Various
        lastName: String, Various
        Sex: Male, Female
        Age: Number or empty
        No of Siblings or Spouses on Board: Integer
        No of Parents or Children on Board: Integer
        Ticket Number: Various
        Passenger Fare: Float
        Cabin: String or Empty
        Port of Embarkation: Southampton, Cherbourg, Queenstown or Empty
        Life Boat: String or Empty
        Survived: No, Yes
        '''
        preview_data(sample, f)

    # Load translated data
    split = functools.partial(xy_split, y_field='Survived')
    Dataloader = functools.partial(
        C_Dataloader, collate_fn=split, **DATALOADER_PARAMS
    )
    train, test = map(Dataloader, from_csv(
        DATA_PATH, 0.2, DEFAULT_TRANS_TABLE))
    models = [Model_1, Model_2, Model_3]
    for i, _ in enumerate(models, 1):
        model = train_model(_(), train, **TRAINING_PARAMS)
        print(f'Model {i} accuracy:', test_model(model, test))
    # Training result
    # Performance: Model1 ~ Model2 > Model3
