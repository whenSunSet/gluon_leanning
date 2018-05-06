from mxnet.gluon import nn

channels_1 = 96
kernel_size_1 = 11
channels_2 = 256
kernel_size_2 = 5
channels_3 = 384
kernel_size_3 = 3
channels_4 = 384
kernel_size_4 = 3
channels_5 = 256
kernel_size_5 = 3

pool_size_1 = (3, 3)
pool_size_2 = (3, 3)
pool_size_3= (3, 3)

net = nn.Sequential()
with net.name_scope():
    #卷积层1：输入 227*227*1，卷积核 11*11*1，步长 4，输出 55*55*96，激活函数 relu
    net.add(nn.Conv2D(channels=channels_1, kernel_size=kernel_size_1, strides=4, activation='relu'))
    #池化层1：输入 55*55*96，池化核 3*3，步长 2，输出 27*27*96
    net.add(nn.MaxPool2D(pool_size=pool_size_1, strides=2))

    #卷积层2：输入 27*27*96，卷积核 5*5*96，步长 1，填充：上下左右各2，输出 27*27*256，激活函数 relu
    net.add(nn.Conv2D(channels=channels_2, kernel_size=kernel_size_2, strides=1, padding=(2,2), activation='relu'))
    #池化层2：输入 27*27*256，池化核 3*3，步长 2，输出 13*13*256
    net.add(nn.MaxPool2D(pool_size=pool_size_2, strides=2))

    #卷积层3：输入 13*13*256，卷积核 3*3*256，步长 1，填充：上下左右各1，输出 13*13*384，激活函数 relu
    net.add(nn.Conv2D(channels=channels_3, kernel_size=kernel_size_3, strides=1, padding=(1,1), activation='relu'))

    #卷积层4：输入 13*13*384，卷积核 3*3*384，步长 1，填充：上下左右各1，输出 13*13*384，激活函数 relu
    net.add(nn.Conv2D(channels=channels_4, kernel_size=kernel_size_4, strides=1, padding=(1,1) ,activation='relu'))

    #卷积层5：输入 13*13*384，卷积核 3*3*384，步长 1，填充：上下左右各1，输出 13*13*256，激活函数 relu
    net.add(nn.Conv2D(channels=channels_5, kernel_size=kernel_size_5, strides=1, padding=(1,1) ,activation='relu'))
    #池化层5：输入 13*13*256，池化核 3*3，步长 2，输出 6*6*256
    net.add(nn.MaxPool2D(pool_size=pool_size_3, strides=2))

    #将卷积层的输出平铺，然后映射为4096，丢弃值为0.5
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(0.5))

    #继续映射为4096，丢弃值为0.5
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(0.5))

    #映射为最终的分类10
    net.add(nn.Dense(10))

import sys
sys.path.append('..')
import util

train_data, test_data = util.load_data_fashion_mnist(batch_size=64, resize=227)

from mxnet import init
from mxnet import gluon

ctx = util.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate': 0.01})
util.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)