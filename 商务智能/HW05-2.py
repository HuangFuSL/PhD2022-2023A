'''
Build an AE model using convolution operation together with maxpooling and upsampling on MINST dataset
'''

import os
import torch
import torch.nn
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

# ! Types and class aliases

C_Tensor = torch.Tensor
C_Array = np.ndarray
C_Module = torch.nn.Module
T_XYPair = T.Tuple[C_Array, C_Array]


def load_data(path: str, to_args: T.Dict[str, T.Any]) -> T.Tuple[T_XYPair, T_XYPair]:
    with np.load(path) as f:
        x_train, y_train = torch.from_numpy(f['x_train']).to(**to_args), f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data(**DATA_PARAMS)
