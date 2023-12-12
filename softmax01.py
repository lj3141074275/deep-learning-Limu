import sys
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
#import d2lutil.common as common

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

## 读取小批量数据
batch_size = 256
trans = transforms.ToTensor()
#train_iter, test_iter = common.load_fashion_mnist(batch_size) #无法翻墙的，可以参考这种方法取下载数据集
mnist_train  = torchvision.datasets.FashionMNIST(#训练集一般是true，要shuffle
    root="../data", train=True, transform=trans, download=True) # 需要网络翻墙，这里数据集会自动下载到项目跟目录的/data目录下
mnist_test  = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True) # 需要网络翻墙，这里数据集会自动下载到项目跟目录的/data目录下
print(len(mnist_train))  # train_iter的长度是235；说明数据被分成了234组大小为256的数据加上最后一组大小不足256的数据
print('11111111')


## 展示部分数据
def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))#每个图片显示大小
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


train_data, train_targets = next(iter(data.DataLoader(mnist_train, batch_size=18)))#在dataloder存储批量大小18
#展示部分训练数据
show_fashion_mnist(train_data[0:10], train_targets[0:10])
'''
训练前可以看一下数据读取有多快，数据瓶颈
train_iter=data.DataLoader(mnist_train,batch_size=256,shuffle=True,num_workers=4)
timer=d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')#时间比沐1.72sec高很多，本机6.98sec
'''

# 初始化模型参数 图片是28×28=784的，所以输入是784，输出是10个不一样的类别
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #784×10个w，输出×输入
b = torch.zeros(num_outputs, requires_grad=True)                              #10个b，输出个数

'''
矩阵相乘运算
x=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(x.sum(0,keepdim=True),x.sum(1,keepdim=True))
[1,7,9]和[6,15]就是按照列向量和dim=1行向量分别相加

softmax场景：
y=torch.tensor([0,2])  2*1
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]) 2*3
y_hat[[0,1],y]             输出样本0和样本1对应真实值的概率，样本0的第0个概率是0.1，样本1的第二个概率式0.5，相当于（0,0）和（1,2）
输出tensor([0.1000,0.5000])
cross_entropy(y_hat, y)之后就是输出样本0和1的损失[2.3026,0.6931]

'''


# 定义模型 使用分数概率转换为全是正小数，指数后按照行求和，再全部除以总数分母partition，每一行就是一张图片
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):#256*784矩阵784*10得到256*10再矩阵10*1得到输出256个结果，就是批量大小18
    return softmax(torch.matmul(X.reshape(-1, num_inputs), W) + b)#x.reshape之后就会变成18×784，就是批量大小×图像大小，预测的y_hat


# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))#交叉熵损失-logYy,Y的标号y,view=reshape


# 计算分类准确率,概率最大的下标找出来为预测y，与真实y比较
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 计算这个训练集的准确率，不使用梯度
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


num_epochs, lr = 5, 0.1


# 本函数已保存在d2lzh包中方便以后使用
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
        print("w误差 ",torch.sum(W), "/n b误差 ", torch.sum(b),'/n')


# 训练模型
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测模型
for X, y in test_iter:
    break
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
'''
测试结果
60000
11111111
epoch 1, loss 0.7859, train acc 0.749, test acc 0.793
epoch 2, loss 0.5703, train acc 0.813, test acc 0.812
epoch 3, loss 0.5258, train acc 0.824, test acc 0.819
epoch 4, loss 0.5010, train acc 0.833, test acc 0.825
epoch 5, loss 0.4858, train acc 0.837, test acc 0.824
epoch 6, loss 0.4738, train acc 0.841, test acc 0.830
epoch 7, loss 0.4649, train acc 0.843, test acc 0.830
epoch 8, loss 0.4580, train acc 0.845, test acc 0.833
epoch 9, loss 0.4520, train acc 0.847, test acc 0.834
epoch 10, loss 0.4468, train acc 0.848, test acc 0.833

https://blog.csdn.net/qq_38473254/article/details/131718450?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170081474416800197082198%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170081474416800197082198&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-131718450-null-null.142^v96^pc_search_result_base1&utm_term=softmax%E5%9B%9E%E5%BD%92%E4%BB%A3%E7%A0%81&spm=1018.2226.3001.4187
详细讲述softmax
'''
