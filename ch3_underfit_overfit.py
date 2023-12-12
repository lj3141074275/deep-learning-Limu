
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
max_degree = 20 # 特征为20就是每一个样本是一个[20,1]的tensor
n_train, n_test = 100, 100 # 100个测试样本、100验证样本
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6]) # 真实标号为5，剩下都是0

features = np.random.normal(size=(n_train+n_test,1))#真随机，没有正态分布
print(features.sum())

#np.random.normal(loc,scale,size）loc:正态分布的均值 scale:正态分布的标准差 size:设定数组形状
print(features.shape)
np.random.shuffle(features)
print(features[:2])
#np.arange([start, ]stop, [step, ]dtype = None)
print(np.arange(max_degree))
print(np.arange(max_degree).reshape(1,-1))
print(np.power([[10,20]],[[1,2]]))#对应元素指数乘法
poly_features = np.power(features, np.arange(max_degree).reshape(1,-1)) # 对第所有维的特征取0次方、1次方、2次方...19次方  
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1) # i次方的特征除以(i+1)阶乘math.gamma函数
labels = np.dot(poly_features,true_w) # 根据多项式生成y，即生成真实的labels
labels += np.random.normal(scale=0.1,size=labels.shape) # 对真实labels加噪音进去
print("真实y")
print(labels.shape)#200个数值，分成2部分，前100个是train，后100个是test


#看一下前两个样本
true_w, features, poly_features, labels = \
    [torch.tensor(x,dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
print(features[:2]) # 前两个样本的x 精度损失
print(poly_features[:2,:]) # 前两个样本的x的所有次方
print(labels[:2])  # 前两个样本的x对应的y 真实的y，对应到main中的

# 实现一个函数来评估模型在给定数据集上的损失
def evaluate_loss(net, data_iter, loss):#train（）会定义好
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2) # 两个数的累加器
    for X, y in data_iter: # 从迭代器中拿出对应特征和标签
        out = net(X)#线性层输出
        y = y.reshape(out.shape) # 将真实标签改为网络输出标签的形式，统一形式
        l = loss(out, y) # 计算网络输出的预测值与真实值之间的损失差值
        metric.add(l.sum(), l.numel()) # 总量除以个数，等于平均
    return metric[0] / metric[1] # 返回数据集的平均损失

# 定义训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    print(input_shape)#4,输入的就是4,输入是2就欠拟合，大于4会过拟合
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 单层线性回归
    batch_size = min(10,train_labels.shape[0])
    #print(train_labels.shape[0])#输出是100
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)    
    trainer = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',
                            xlim=[1,num_epochs],ylim=[1e-3,1e2],legend=['train','test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net,test_iter,loss)))
    print('weight',net[0].weight.data.numpy()) # 训练完后打印，打印最终学到的weight值  

if __name__ =='__main__':
    print(labels[:n_train])#100个数值
    print('---')
    print(labels[n_train:])
    print('---')
    print(labels.shape)
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])#三阶多项式，打印四个weight
    # 相当于用一阶多项式拟合真实的三阶多项式，欠拟合,输出2个weight
    #train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    # 十九阶多项式函数拟合(过拟合)
    # 相当于用十九阶多项式拟合真实的三阶多项式，过拟合，效果好像还不错，labels是200数组。
    #train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])

    plt.show()