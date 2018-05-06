from mxnet import nd
def pure_batch_norm(X, gama, beta, epa=1e-5):
    if(len(X.shape) == 2):
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    else:
        mean = X.mean(axis=(0, 2, 3), keepdims = True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
    X_hat = (X - mean) / nd.sqrt(variance + epa)
    return gama.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)


A = nd.arange(6).reshape((3, 2))
print(A)
print(pure_batch_norm(A, gama=nd.array([1,1]), beta=nd.array([0, 0])))

B = nd.arange(18).reshape((1,2,3,3))
print(B)

print(pure_batch_norm(B, gama=nd.array([1,1]), beta=nd.array([0, 0])))

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

def batch_norm(X, gamma, beta, is_training, moving_mean, moving_variance,
               eps = 1e-5, moving_momentum = 0.9):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确的广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        # 变形使得可以正确的广播
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    # 均一化
    if is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)
        #!!! 更新全局的均值和方差
        moving_mean[:] = moving_momentum * moving_mean + (
                1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (
                1.0 - moving_momentum) * variance
    else:
        #!!! 测试阶段使用全局的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)

    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

weight_scale = .01

# 输出通道 = 20, 卷积核 = (5,5)
c1 = 20
W1 = nd.random.normal(shape=(c1,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(c1, ctx=ctx)

# 第1层批量归一化
gamma1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(c1, ctx=ctx)
moving_variance1 = nd.zeros(c1, ctx=ctx)

# 输出通道 = 50, 卷积核 = (3,3)
c2 = 50
W2 = nd.random_normal(shape=(c2,c1,3,3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(c2, ctx=ctx)

# 第2层批量归一化
gamma2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(c2, ctx=ctx)
moving_variance2 = nd.zeros(c2, ctx=ctx)

# 输出维度 = 128
o3 = 128
W3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(o3, ctx=ctx)

# 输出维度 = 10
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

# 注意这里moving_*是不需要更新的
params = [W1, b1, gamma1, beta1,
          W2, b2, gamma2, beta2,
          W3, b3, W4, b4]

for param in params:
    param.attach_grad()

def net(X, is_training=False, verbose=False):
    X = X.as_in_context(W1.context)
    X = X.reshape((50, 1, 28, 28))
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=c1)
    ### 添加了批量归一化层
    h1_bn = batch_norm(h1_conv, gamma1, beta1, is_training,
                       moving_mean1, moving_variance1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=c2)
    ### 添加了批量归一化层
    h2_bn = batch_norm(h2_conv, gamma2, beta2, is_training,
                       moving_mean2, moving_variance2)
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net).asscalar()
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        (train_acc/len(train_data)).asscalar(), test_acc))
