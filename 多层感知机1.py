from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from mxnet import autograd

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

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

data, label = mnist_train[0]
print('shape:', data.shape, 'label:', label)

batch_size = 50
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 28 * 28
num_hidden1 = 256
num_hidden2 = 128
num_outputs = 10
weight_scale = .01
w1 = nd.random_normal(shape=(num_inputs, num_hidden1), scale=weight_scale)
b1 = nd.random_normal(shape=(num_hidden1))
w2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
b2 = nd.random_normal(shape=(num_hidden2))
w3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weight_scale)
b3 = nd.random_normal(shape=(num_outputs))
params = [w1, b1, w2, b2, w3, b3]
for param in params:
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    return nd.dot(nd.dot(relu(nd.dot(X.reshape((-1, num_inputs)), w1) + b1), w2) + b2, w3) + b3

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

lr = 0.1
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()

        SGD(params, lr/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net).asscalar()
    print("Epoch %d.Test acc %f" % (epoch, test_acc))


data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))

