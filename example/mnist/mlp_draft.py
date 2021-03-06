# pylint: skip-file
from data import mnist_iterator
import mxnet as mx
import logging

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
fc4 = mx.symbol.FullyConnected(data = act2, name='fc4', num_hidden=10)
mlp1 = mx.symbol.Softmax(data = fc3, name = 'mlp1')
mlp2 = mx.symbol.Softmax(data = mlp1, name = 'mlp2')
mlp=mlp2
#draw network

batch_size = 100
data_shape = (batch_size, 784)
dot = mx.viz.plot_network(mlp, shape={"data":data_shape})
dot.render('test-output/round-table.gv', view=True)






# data

train, val = mnist_iterator(batch_size=100, input_shape = (784,))

# train

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx = mx.cpu(), symbol = mlp, num_round = 20,
    learning_rate = 0.1, momentum = 0.9, wd = 0.00001)

model.fit(X=train, eval_data=val)
