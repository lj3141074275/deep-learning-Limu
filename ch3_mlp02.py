import torch
from torch import nn
from d2l import torch as d2l
#多层感知机
import matplotlib.pyplot as plt
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#第一层输入，第二次隐藏，第三次输出，具有单隐藏层的多层感知机，包含256分隐藏单元
num_inputs,num_outputs,num_hiddens=784,10,256
W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True))#第一层的W1，随机的，784*256
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))#
W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))#256*10
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params=[W1,b1,W2,b2]

def relu(x):
    a=torch.zeros_like(x)#生成和x类型一样的全是0的a
    return torch.max(x,a)

def net(x):
    x=x.reshape(-1,num_inputs)#256*784,批量大小*784，@是矩阵乘法
    h=relu(x @ W1 + b1)#不用转置，看形状，有两层nn
    return (h @ W2 + b2)
loss=nn.CrossEntropyLoss()

num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
plt.show()
print('loss %.4f' % (loss))