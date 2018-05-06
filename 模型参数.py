from mxnet import init, gluon, nd
from mxnet.gluon import nn
import sys

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(4)
            self.output = nn.Dense(2)

    def forward(self, x):
        return self.output(nd.relu(self.hidden(x)))


x = nd.random.uniform(shape=(3, 5))

my_param = gluon.Parameter("good_param", shape=(2,3))
my_param.initialize()
print('data: ', my_param.data(), '\ngrad: ', my_param.grad(),'\nname: ', my_param.name)

net = MLP()
net.initialize()
net(x)

w = net.hidden.weight
b = net.hidden.bias
print('hidden layer name: ', net.hidden.name, '\nweight: ', w, '\nbias: ', b)
print('weight:', w.data(), '\nweight grad:', w.grad(), '\nbias:', b.data(),'\nbias grad:', b.grad())

params = net.collect_params()
print(params)
print(params['mlp0_dense0_bias'].data())
print(params.get('dense0_bias').data())

params = net.collect_params()
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print('hidden weight: ', net.hidden.weight.data(), '\nhidden bias: ',
      net.hidden.bias.data(), '\noutput weight: ', net.output.weight.data(),
      '\noutput bias: ',net.output.bias.data())

net.hidden.bias.initialize(init=init.One(), force_reinit=True)

class MyInit(init.Initializer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._verbosity = True

    def _init_weight(self, name, arr):
        nd.random.uniform(low=10, high=20, out=arr)

net = MLP()
net.initialize(MyInit())
net(x)
print(net.hidden.weight.data())
print(net.hidden.bias.data())

w = net.output.weight
w.set_data(nd.ones(w.shape))
print('output layer modified weight:', net.output.weight.data())

net = MLP()
print(net.collect_params())

net.initialize()
print(net.collect_params())


print(x)
net(x)
print(net.collect_params())


net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu', params=net[1].params))
    net.add(nn.Dense(2, activation='relu'))

net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())

class MLP_SHARE(nn.Block):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)
        with self.name_scope():
            self.hidden1 = nn.Dense(4, activation='relu')
            self.hidden2 = nn.Dense(4, activation='relu')
            self.hidden3 = nn.Dense(4, activation='relu', params=self.hidden2.params)
            self.output = nn.Dense(2)

    def forward(self, *args):
        return self.output(self.hidden3(self.hidden2(self.hidden1(x))))

net = MLP_SHARE()
net.initialize()
net(x)
print(net.hidden2.weight.data())
print(net.hidden3.weight.data())
print(net.params)
print(net.collect_params())