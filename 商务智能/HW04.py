import functools
import os
import re
import typing as T

import gensim.models
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data.dataloader

os.chdir(os.path.dirname(__file__))

# !Types, class aliases and function aliases
T_DATA = T.Sequence[T.Any]
T_TRANSLATION = T.Optional[T.Callable[[T.Any], T.Any]]
T_TRANS_TABLE = T.Optional[T.Mapping[str, T_TRANSLATION]]
C_Dataloader = torch.utils.data.dataloader.DataLoader
C_Word2Vec = gensim.models.Word2Vec
C_KeyedVectors = gensim.models.keyedvectors.KeyedVectors
C_Module = torch.nn.Module
F_Split = sklearn.model_selection.train_test_split
F_Pad = torch.nn.utils.rnn.pad_sequence

# ! Constants
# ^ Data
REVIEW_PATH = 'products_reviews2.csv'
CONTENT_PATH = 'reviews_text2.csv'
CONTENT_KEY = 'review/text'
JOINT_KEY = 'index'
SCORE_KEY = 'review/score'
# ^ Preprocess
PUNC_RE = re.compile(r'[.,!?;:()\[\]{}\-&/*\+=_@><#\^\\|]')
WV_PARAMS = {
    'min_count': 5,
    'vector_size': 128,
    'workers': 24,
    'epochs': 10
}
TEST_RATIO = 0.2
# ^ Training
TRAIN_PARAMS = {
    'epochs': 10,
    'lr': 0.001,
    'batch_size': 128,
    'optim': torch.optim.SGD,
    'loss': torch.nn.CrossEntropyLoss
}
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data() -> T.Tuple[T.List[str], T.List[int]]:
    review = pd.read_csv(REVIEW_PATH)
    content = pd.read_csv(CONTENT_PATH)
    joint = pd.merge(review, content, on=JOINT_KEY)
    ret = joint[CONTENT_KEY].to_list(), joint[SCORE_KEY].to_list()
    return ret


def to_embed(embed: C_KeyedVectors, data: T.List[str]):
    for _ in data:
        try:
            yield embed[_]
        except KeyError:
            continue


def preprocess_data(*data: T_DATA):
    X, y = data
    y_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    X = [re.sub(PUNC_RE, ' ', x).lower().split() for x in X]
    y = [y_map[_] for _ in y]
    aX, bX, ay, by = F_Split(X, y, test_size=TEST_RATIO)
    embed = C_Word2Vec(aX, **WV_PARAMS)
    aXv = to_embed(embed.wv, aX)
    bXv = to_embed(embed.wv, bX)
    return list(zip(aXv, ay)), list(zip(bXv, by))


def collate_fn(batch):
    X, y = zip(*batch)
    X = F_Pad([torch.tensor(_) for _ in X], padding_value=0).to(DEV)
    y = torch.tensor(y, dtype=torch.long).to(DEV)
    return X, y


class Model(C_Module):

    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(128, 128)
        self.linear = torch.nn.Linear(128, 3)

    def forward(self, x):
        o, h = self.rnn(x)
        x = self.linear(torch.sigmoid(h.reshape(-1, 128)))
        return torch.softmax(x, dim=1)


def train(model: C_Module, data: C_Dataloader, **params: T.Any):
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


def test_model(model: C_Module, data: C_Dataloader) -> float:
    model.eval()
    ret = torch.tensor([], dtype=torch.float32, requires_grad=False, device=DEV)
    for x, y in data:
        y_pred = model(x)
        ret = torch.concat([ret, (torch.argmax(y_pred, dim=1) == y)])
    return ret.to('cpu').mean().item()


if __name__ == '__main__':
    dataset = preprocess_data(*load_data())
    loader = functools.partial(
        C_Dataloader, batch_size=TRAIN_PARAMS['batch_size'], collate_fn=collate_fn
    )
    train_loader, test_loader = map(loader, dataset)
    model = Model().to(DEV)
    model = train(model, train_loader, **TRAIN_PARAMS)
    print(test_model(model, test_loader))
