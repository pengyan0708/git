<<<<<<< HEAD
from __future__ import print_function

from mxnet import  gluon




def get_cnn(num_outputs=10):
    # define the architecture of the CNN
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
=======
import torch
import torch.nn as nn

def get_cnn(num_outputs=10):
    # define the architecture of the CNN
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(50 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, num_outputs)
    )
    return cnn


def get_deep_cnn(num_outputs=10):
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 100),
        nn.ReLU(),
        nn.Linear(100, num_outputs)
    )
    return cnn

def get_dropout_cnn(num_outputs=10, dropout_rate=0.5):
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(50 * 7 * 7, 100),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(100, num_outputs)
    )
    return cnn

def get_diff_kernel_cnn(num_outputs=10):
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, num_outputs)
    )
    return cnn

def get_bn_cnn(num_outputs=10):
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding=1),
        nn.BatchNorm2d(30),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, padding=1),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(50 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, num_outputs)
    )
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    return cnn

def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
<<<<<<< HEAD
    else:
        raise NotImplementedError
=======
    elif net_type == 'cnn1':
        net = get_deep_cnn(num_outputs)
    elif net_type == 'cnn2':
        net = get_bn_cnn(num_outputs)
    elif net_type == 'cnn3':
        net = get_diff_kernel_cnn(num_outputs)
    elif net_type == 'cnn4':
        net = get_dropout_cnn(num_outputs)
    else:
        raise NotImplementedError("Unsupported network type")
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    return net