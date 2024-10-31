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
    return cnn

def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    else:
        raise NotImplementedError
    return net