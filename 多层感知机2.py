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
num_hidden3 = 64
num_hidden4 = 32
num_outputs = 10

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_hidden1, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden2, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden3, activation="relu"))
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

lr = 0.1
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
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
    print("Epoch %d.Test acc %f" % (epoch, test_acc))


data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
help(nd.Activation)

