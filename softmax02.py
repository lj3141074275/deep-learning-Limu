import sys

import torch
from d2l import torch as d2l
from torch import nn

## 读取小批量数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print(len(train_iter))  # train_iter的长度是235；说明数据被分成了234组大小为256的数据加上最后一组大小不足256的数据
print('11111111')
#for X, y in train_iter:
    #print(X, y)
   # break  # 尝试打印第一组X, y的形状：torch.Size([256, 1, 28, 28])  torch.Size([256])

# 定义模型  pytorch不会隐式调整输入的形状，先定义展平flatten=reshape，在线性层前调整网络输入形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))#输入784，输出10


# 初始化参数 如果是线性层
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weight)

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5

print('123')
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 计算这个训练集的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

print('66')
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 执行优化方法
            if optimizer is not None:
                optimizer.step()
            else:
                d2l.sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

print('55')
# 开始训练模型
if __name__=='__main__':
    train_ch3(net, train_iter, test_iter, loss, num_epochs, 256,None, 0.3, optimizer)

'''
epoch 1, loss 0.0120, train acc 0.764, test acc 0.814
epoch 2, loss 0.0236, train acc 0.797, test acc 0.758
epoch 3, loss 0.0367, train acc 0.787, test acc 0.753
epoch 4, loss 0.0501, train acc 0.795, test acc 0.753
epoch 5, loss 0.0578, train acc 0.806, test acc 0.765
准确度下降
'''