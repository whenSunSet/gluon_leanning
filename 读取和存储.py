from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
y = nd.zeros(4)
filename = "test1.params"
nd.save(filename, [x,y])

a, b = nd.load(filename)
print(a, b)


mydict = {"x":x, "y":y}
filename = "test2.params"
nd.save(filename, mydict)
c = nd.load(filename)
print(c)

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(2))
    return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
print(net(x))

filename = "mlp.params"
net.save_params(filename)

import mxnet as mx
net2 = get_net()
net2.load_params(filename, mx.cpu(0))
print(net2(x))

