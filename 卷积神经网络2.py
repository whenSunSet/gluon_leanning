from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )


from mxnet import gluon
import sys
sys.path.append('..')

# 初始化
net.initialize()

# 获取数据
batch_size = 256

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
# 训练
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        if(data.shape[0] != batch_size):
            break
        output = net(data.reshape((batch_size, 1, 28, 28)))
        acc += accuracy(output, label)
    return acc / len(data_iterator)

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label)

from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from mxnet import autograd

lr = 0.1
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        if(data.shape[0] != batch_size):
            break
        with autograd.record():
            output = net(data.reshape((batch_size, 1, 28, 28)))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net).asscalar()
    print("Epoch %d.Test acc %f" % (epoch, test_acc))

