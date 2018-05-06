from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):

    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)
        with self.name_scope():
            self.hidden = nn.Dense(256, activation="relu")
            self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2, 20))
print(net.forward(x))
print('hidden layer name with default prefix:', net.hidden.name)
print('output layer name with default prefix:', net.output.name)

class MLP_NO_NAMESCOPE(nn.Block):
    def __init__(self, **kwargs):
        super(MLP_NO_NAMESCOPE, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP_NO_NAMESCOPE(prefix='my_mlp_')
print('hidden layer name without prefix:', net.hidden.name)
print('output layer name without prefix:', net.output.name)

class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x

nets = MySequential()
with nets.name_scope():
    nets.add(nn.Dense(256, activation='relu'))
    nets.add(nn.Dense(10))
nets.initialize()
nets(x)


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = nd.random_uniform(shape=(10, 20))
        with self.name_scope():
            self.dense = nn.Dense(10, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight) + 1)
        x = self.dense(x)
        return x

net = FancyMLP()
net.initialize()
net(x)

class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(64, activation='relu'))
            self.net.add(nn.Dense(32, activation='relu'))
            self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP())
net.add(nn.Dense(10))
net.initialize()
print(net(x))