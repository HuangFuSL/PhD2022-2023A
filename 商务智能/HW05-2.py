'''
Build an AE model using convolution operation together with maxpooling and upsampling on MINST dataset
'''

import os
import torch
import functools
import matplotlib.pyplot
import torch.nn
import torch.optim
import torch.utils.data
import typing as T

import numpy as np

os.chdir(os.path.dirname(__file__))

# ! Constants

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PARAMS = {
    'path': 'mnist.npz',
    'to_args': {
        'device': DEV,
        'non_blocking': True
    }
}
LOADER_PARAMS = {
    'batch_size': 1024,
    'shuffle': True
}
TRAIN_CLASSIFIER_PARAMS = {
    'epochs': 40,
    'lr': 0.002,
    'optim': torch.optim.Adam,
    'loss': torch.nn.CrossEntropyLoss
}
TRAIN_AE_PARAMS = {
    'epochs': 100,
    'lr': 0.001,
    'optim': torch.optim.Adam,
    'loss': torch.nn.MSELoss
}
EXPORT_PARAMS = {
    'path': 'export',
    'num': 100,
    'start': 0
}

# ! Types and class aliases

C_Array = np.ndarray
C_DataLoader = torch.utils.data.DataLoader
C_Module = torch.nn.Module
C_Tensor = torch.Tensor
T_XYPair = T.List[T.Tuple[C_Tensor, C_Tensor]]


def add_noise(x: C_Tensor, p: float):
    return x + torch.randn_like(x, device=x.device) * (p ** 0.5)


def load_data(path: str, to_args: T.Dict[str, T.Any]) -> T.Tuple[T_XYPair, ...]:
    with np.load(path) as f:
        result: T.Dict[str, C_Tensor] = {}
        for _ in f:
            if 'x' in _:
                result[_] = torch.from_numpy(f[_] / 255).float().to(**to_args)
            else:
                result[_] = torch.from_numpy(f[_]).to(**to_args)
        return \
            list(zip(result['x_train'], result['y_train'])), \
            list(zip(result['x_test'], result['y_test'])), \
            list(zip(add_noise(result['x_train'], 0.2), result['x_train'])), \
            list(zip(add_noise(result['x_test'], 0.2), result['y_test']))


class PooledConv2d(C_Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool_size)
        )

    def forward(self, x: C_Tensor) -> C_Tensor:
        return self.layers(x)


class PooledTransposedConv2d(C_Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rescale_size: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU(),
            torch.nn.Upsample(mode='bicubic', size=rescale_size)
        )

    def forward(self, x: C_Tensor) -> C_Tensor:
        return self.layers(x)


class Encoder(C_Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            PooledConv2d(1, 16, 3, 2),
            PooledConv2d(16, 32, 3, 2)
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(32 * 5 * 5, 256)

    def forward(self, x: C_Tensor) -> C_Tensor:
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(self.dropout(x))


class Decoder(C_Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 24 * 5 * 5)
        self.dropout = torch.nn.Dropout(0.5)
        self.layers = torch.nn.Sequential(
            PooledTransposedConv2d(24, 16, 3, 14),
            PooledTransposedConv2d(16, 8, 3, 21),
            PooledTransposedConv2d(8, 1, 3, 28),
        )

    def forward(self, x: C_Tensor) -> C_Tensor:
        x = self.fc(x)
        x = x.reshape(x.shape[0], 24, 5, 5)
        return self.layers(self.dropout(x))


class AE(C_Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: C_Tensor) -> C_Tensor:
        x = x.unsqueeze(1)
        return self.decoder(self.encoder(x)).squeeze(1)


class Classifier(C_Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x: C_Tensor) -> C_Tensor:
        x = x.unsqueeze(1)
        return self.fc(self.dropout(self.encoder(x)))


def train(model: C_Module, data: C_DataLoader, **params: T.Any):
    model.train()
    optimizer = params['optim'](model.parameters(), lr=params['lr'])
    loss_fn = params['loss']()
    for _ in range(params['epochs']):
        l = 0
        for x, y in data:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            l += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {_ + 1}/{params["epochs"]}: {l/len(data)}')
    return model


def test(*models: T.Callable[[C_Tensor], C_Tensor], data: C_DataLoader) -> float:
    for model in models:
        model.eval()
    ret = torch.tensor(
        [], dtype=torch.float32, requires_grad=False, device=DEV
    )
    for x, y in data:
        y_pred = x
        for model in models:
            y_pred = model(y_pred)
        ret = torch.concat([ret, (torch.argmax(y_pred, dim=1) == y)])
    return ret.to('cpu').mean().item()


def train_classifier(train_loader: C_DataLoader, test_loader: C_DataLoader, device: torch.device, **args):
    model = Classifier().to(device)
    model = train(model, train_loader, **args)
    model.eval()
    acc = test(model, data=test_loader)
    print(f'Classifier trained with {acc:.2%} accuracy')
    return model


def export_image(model: C_Module, data: C_DataLoader, **kwargs: T.Any):
    model.eval()
    if not os.path.exists(kwargs['path']):
        os.mkdir(kwargs['path'])
    for X, _ in data:
        X2 = X.clone().detach().cpu().numpy()
        Y = model(X)
        Y2 = Y.clone().detach().cpu().numpy()

        for i, (x2, y2) in enumerate(zip(X2, Y2), start=kwargs['start']):
            fig, (ax0, ax1) = matplotlib.pyplot.subplots(1, 2)  # type: ignore
            ax0.imshow(x2, cmap='gray')
            ax1.imshow(y2, cmap='gray')
            matplotlib.pyplot.savefig(
                os.path.join(kwargs['path'], f'{i}.png')
            )  # type: ignore
            matplotlib.pyplot.close(fig)
            if i >= kwargs['num']:
                return


if __name__ == '__main__':
    DataLoader = functools.partial(C_DataLoader, **LOADER_PARAMS)
    train_loader, test_loader, ae_train_loader, ae_test_loader = map(
        DataLoader, load_data(**DATA_PARAMS))
    classifier = train_classifier(
        train_loader, test_loader, DEV, **TRAIN_CLASSIFIER_PARAMS
    )
    model = AE().to(DEV)
    train(model, data=ae_train_loader, **TRAIN_AE_PARAMS)
    test_acc = test(model, classifier, data=ae_test_loader)
    print(f'Auto Encoder trained with {test_acc:.2%} accuracy')
    export_image(model, ae_test_loader, **EXPORT_PARAMS)
