import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
#函数 y=0.05+sum(0.01*x1+0.01*x2+0.01*x3........)+噪音（均值为0，方差为0.01）
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5 # 数据越简单，模型越复杂，越容易过拟合。num_inputs为特征维度
true_w, true_b = torch.ones((num_inputs,1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train) # 生成人工数据集,Ctrl+函数，可以跳进去看函数具体干啥，这就是生成真实的y
#print(train_data)

train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数
def init_params():
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)#函数torch.normal(mean=1,std=2,size=(3,4))
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

# 定义L2范数惩罚，就是把W平方除以2
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
#return torch.sum(torch.abs(w)) L1范数

# 定义训练函数
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss#lambda，相当于定义了一个函数
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[5,num_epochs],legend=['train','test'])#动画效果
    for epoch in range(num_epochs):
        for X, y in train_iter:
            #with torch.enable_grad():
            l = loss(net(X),y) + lambd * l2_penalty(w)#区别，加上超参数*L2
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if(epoch+1) % 5 == 0:
            if(epoch+1) % 5 ==0:
                animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net,test_iter,loss)))
    print('w的L2范数是',torch.norm(w).item())

'''
简洁实现：1、建立网络。2、损失函数。3、优化器（根据反向传播求得梯度,用优化器更新参数）。4、从训练集去除数据，进行训练，

def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss=nn.MSELoss()
    num_epochs,lr=100,0.003
    trainer=torch.optim.SGD([{
        "params":net[0].weight,
        "weight_decay":wd,},{#就是lambd
        "params":net[0].bias}],lr=lr)
    animator=d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[5,num_epochs],legend=['train','test'])#动画效果
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l=loss(net(X),y)
            l.backward()
            trainer.step()
'''

# 无正则化直接训练
#train(lambd=0)  #训练集小，过拟合，测试集损失不下降,test一条直线了，w的L2范数是13.392132759094238，两条线差距越来越大
# 使用权重衰退
#train(lambd=3)# 0.3826645016670227，还是偏大，可以调大lamda
train(lambd=10)# 0.02332453615963459
train(5)# 0.04332248866558075
plt.show()