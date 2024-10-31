<<<<<<< HEAD
from __future__ import print_function
from mxnet import nd, autograd, gluon
import numpy as np
import random
import mxnet as mx




def get_shapes(dataset):
    # determine the input/output shapes
    if dataset == 'FashionMNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'MNIST':
=======
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def get_shapes(dataset):
    # determine the input/output shapes
    if dataset in ['FashionMNIST', 'MNIST']:
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset))
    return num_inputs, num_outputs, num_labels

<<<<<<< HEAD

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,
                                              shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250,
                                             shuffle=False, last_batch='rollover')
    elif dataset == 'MNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000,
                                              shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 250,
                                             shuffle=False, last_batch='rollover')
=======
def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True),
            batch_size=64, shuffle=True)
        test_data = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True),
            batch_size=64, shuffle=False)
    elif dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True),
            batch_size=64, shuffle=True)
        test_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True),
            batch_size=64, shuffle=False)
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset))
    return train_data, test_data

<<<<<<< HEAD

def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST",
                seed=1):
=======
def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1):
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

<<<<<<< HEAD
    # assign training data to each worker
=======
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    server_data = []
    server_label = []

<<<<<<< HEAD
    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
            elif dataset == "MNIST":
                x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
            else:
                raise NotImplementedError("Dataset {} not implemented".format(dataset))

            y = y.as_in_context(ctx)

            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    server_data = nd.concat(*server_data, dim=0)
    server_label = nd.concat(*server_label, dim=0)

    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

    # randomly permute the workers
=======
    server_counter = [0 for _ in range(num_labels)]
    for data, label in train_data:
        for x, y in zip(data, label):
            x = x.to(ctx)  # 将数据移动到指定的计算设备上
            y = y.to(ctx)

            upper_bound = (y.item()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.item()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()

            if server_counter[int(y.item())] < server_pc:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.item())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + np.floor(rd * worker_per_group))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    # 将列表中的元素连接为张量
    server_data = torch.stack(server_data, dim=0)
    server_label = torch.stack(server_label, dim=0)

    each_worker_data = [torch.stack(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.stack(each_worker, dim=0) for each_worker in each_worker_label]

    # 随机排列每个worker的数据和标签
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label
<<<<<<< HEAD




=======
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
