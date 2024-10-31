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
    return cnn

def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
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
    return net