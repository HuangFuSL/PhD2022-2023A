'''
For the amazon review dataset, build a model based on CNN to predict the review score, given the review text and other information you could use.
'''

import functools
import itertools
import os
import re
import typing as T

import gensim.models
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.utils.rnn
import torch.optim
import torch.utils.data.dataloader
import torch.utils.data.dataset
import typing_extensions as TE

os.chdir(os.path.dirname(__file__))

# ! Constants

PUNC_RE = re.compile(r'[.,!?;:()\[\]{}\-&/*\+=_@><#\^\\|]')
SCORE_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
TEXT_KEY = 'review/text'
SCORE_KEY = 'review/score'
JOINT_KEY = 'index'
DATA_PARAMS = {
    'path': {
        'score': 'products_reviews2.csv',
        'text': 'reviews_text2.csv'
    },
    'test_size': 0.2
}
LOADER_PARAMS = {
    'batch_size': 32,
    'shuffle': True
}
WV_PARAMS = {
    'min_count': 5,
    'vector_size': 128,
    'workers': 8,
    'epochs': 10
}
TRAIN_PARAMS = {
    'epochs': 10,
    'lr': 0.001,
    'optim': torch.optim.SGD,
    'loss': torch.nn.CrossEntropyLoss
}
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ! Types and class aliases

T_RECORD = T.Mapping[str, T.Any]
T_RECORDS = T.Sequence[T_RECORD]
C_Module = torch.nn.Module
C_Array = np.ndarray
C_Tensor = torch.Tensor
C_DataLoader = torch.utils.data.dataloader.DataLoader
C_Word2Vec = gensim.models.Word2Vec
C_KeyedVectors = gensim.models.keyedvectors.KeyedVectors
F_split = sklearn.model_selection.train_test_split
F_pad = torch.nn.utils.rnn.pad_sequence
C_Chain = itertools.chain


def load_data(path, **kwargs) -> T.Tuple[T_RECORDS, T_RECORDS]:
    s = pd.read_csv(path['score'])
    t = pd.read_csv(path['text'])
    result = pd.merge(s, t, on=JOINT_KEY)[[SCORE_KEY, TEXT_KEY]]
    train, test = F_split(result.to_dict(orient='records'), **kwargs)
    return train, test


def to_embed(embed: C_KeyedVectors, data: T.List[T.List[str]]):
    for _ in data:
        try:
            yield embed[_]
        except KeyError:
            continue


def embed(train: T_RECORDS, test: T_RECORDS):
    def get_text(x): return [PUNC_RE.sub(
        '', _[TEXT_KEY]).lower().split() for _ in x]

    def get_score(x): return [SCORE_MAP[_[SCORE_KEY]] for _ in x]
    def split(x): return (get_text(x), get_score(x))

    train_text, train_score = split(train)
    test_text, test_score = split(test)

    model = C_Word2Vec(train_text, **WV_PARAMS)
    train_embed = to_embed(model.wv, train_text)
    test_embed = to_embed(model.wv, test_text)

    return list(zip(train_embed, train_score)), list(zip(test_embed, test_score))


def collate_fn(batch: T.Sequence[T.Tuple[C_Array, int]], device: torch.device):
    x, y = zip(*batch)
    x = F_pad([torch.from_numpy(_) for _ in x]).permute(1, 2, 0)
    y = torch.tensor(y)

    return x.to(device), y.to(device)


class PooledConv(C_Module):

    def __init__(
        self,
        conv_args: T.Optional[T.Dict[str, T.Any]] = None,
        pool_args: T.Optional[T.Dict[str, T.Any]] = None
    ):
        super().__init__()
        if conv_args is None:
            conv_args = {}
        if pool_args is None:
            pool_args = {}
        self.conv = torch.nn.Conv1d(**conv_args)
        self.relu = torch.nn.ReLU()
        self.pool_args = pool_args

    def forward(self, x: C_Tensor) -> C_Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1], **self.pool_args)
        return x.squeeze(2)


class Model(C_Module):

    def __init__(
            self,
            categories: int,
            in_channels: int, out_channels: int, kernel_sizes: T.List[int]
    ):
        super().__init__()
        self.convs = []
        for _ in kernel_sizes:
            conv_args = {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': _
            }
            new_layer = PooledConv(conv_args)
            self.convs.append(new_layer)

        self.fc = torch.nn.Linear(out_channels*len(kernel_sizes), categories)

    def forward(self, x: C_Tensor):
        results = [_(x) for _ in self.convs]
        concated = torch.concat(results, dim=1)
        return F.softmax(self.fc(concated), dim=1)


def train(model: C_Module, data: C_DataLoader, **params: T.Any):
    model.train()
    optimizer = params['optim'](model.parameters(), lr=params['lr'])
    loss_fn = params['loss']()
    for _ in range(params['epochs']):
        print(f'Epoch {_ + 1}/{params["epochs"]}')
        for x, y in data:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
    return model


def test_model(model: C_Module, data: C_DataLoader) -> float:
    model.eval()
    ret = torch.tensor(
        [], dtype=torch.float32, requires_grad=False, device=DEV
    )
    for x, y in data:
        y_pred = model(x)
        ret = torch.concat([ret, (torch.argmax(y_pred, dim=1) == y)])
    return ret.to('cpu').mean().item()


if __name__ == '__main__':
    collate = functools.partial(collate_fn, device=DEV)
    DataLoader = functools.partial(
        C_DataLoader, collate_fn=collate, **LOADER_PARAMS)
    train_data, test_data = map(DataLoader, embed(*load_data(**DATA_PARAMS)))
    model = Model(3, 128, 64, [3, 4, 5, 6]).to(DEV)
    model = train(model, train_data, **TRAIN_PARAMS)
    print(test_model(model, test_data))
