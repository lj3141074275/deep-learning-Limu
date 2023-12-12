import os
import sys

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from torch import nn


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),#3D的需要展平flatten后，线性层+relu
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
plt.show()