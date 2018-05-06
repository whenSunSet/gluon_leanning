from mxnet import nd
w = nd.arange(4).reshape((1, 1, 2, 2))
b = nd.array([1])
data = nd.arange(9).reshape((1, 1, 3, 3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1])
print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1], stride=(2,2), pad=(1,1))
print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)

w = nd.arange(8).reshape((1, 2, 2, 2))
data = nd.arange(18).reshape((1, 2, 3, 3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)

w = nd.arange(16).reshape((2, 2, 2, 2))
data = nd.arange(18).reshape((1, 2, 3, 3))
b = nd.array([1,2])
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)

data = nd.arange(18).reshape((1, 2, 3, 3))
max_pool = nd.Pooling(data=data, pool_type="max", kernel=[2,2])
avg_pool = nd.Pooling(data=data, pool_type="avg", kernel=[2,2])
print('data:', data, '\n\nmax pooling:', max_pool, '\n\navg pooling:', avg_pool)

from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from mxnet import autograd
import mxnet as mx

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

try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
print(ctx)

weight_scale = 0.01
w1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(w1.shape[0], ctx=ctx)
w2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(w2.shape[0], ctx=ctx)
w3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(w3.shape[1], ctx=ctx)
w4 = nd.random_normal(shape=(128, 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(w4.shape[1], ctx=ctx)

params = [w1, b1, w2, b2, w3, b3, w4, b4]
for param in params:
    param.attach_grad()

def net(X, verbose=False):
    X = X.reshape((batch_size, 1, 28, 28))
    out1 = nd.Convolution(data=X, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=w1.shape[0])
    out2 = nd.relu(out1)
    out3 = nd.Pooling(data=out2, pool_type="max", kernel=(2,2), stride=(2,2))
    out4 = nd.Convolution(data=out3, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=w2.shape[0])
    out5 = nd.relu(out4)
    out6 = nd.Pooling(data=out5, pool_type="max", kernel=(2,2), stride=(2,2))
    out7 = nd.dot(nd.flatten(out6), w3) + b3
    out8 = nd.relu(out7)
    out9 = nd.dot(out8, w4) + b4
    if verbose:
        print('1st conv block:', out3.shape)
        print('2nd conv block:', out5.shape)
        print('2nd conv block:', out6.shape)
        print('2nd conv block:', out7.shape)
        print('1st dense:', out8.shape)
        print('2nd dense:', out9.shape)
        # print('output:', out9)
    return out9

for data, _ in train_data:
    net(data, verbose=True)
    break

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

from mxnet import autograd as autograd
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net).asscalar()
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        (train_acc/len(train_data)).asscalar(), test_acc))

data, label = mnist_test[0:50]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))

