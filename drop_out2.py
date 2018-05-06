from mxnet import nd
from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
from mxnet import autograd

import sys
sys.path.append('..')
batch_size = 256
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 28*28
num_outputs = 10

num_hidden1 = 256
num_hidden2 = 256
weight_scale = .01

net = gluon.nn.Sequential()
drop_prob1 = 0.2
drop_prob2 = 0.5
#
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_hidden1, activation="relu"))
    net.add(gluon.nn.Dropout(drop_prob1))
    net.add(gluon.nn.Dense(num_hidden2, activation="relu"))
    net.add(gluon.nn.Dropout(drop_prob2))
    net.add(gluon.nn.Dense(num_outputs))

net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label)


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = .5

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net).asscalar()
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        (train_acc/len(train_data)).asscalar(), test_acc))