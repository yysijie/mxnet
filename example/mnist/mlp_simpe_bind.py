# pylint: skip-file
from data import mnist_iterator
import mxnet as mx
import logging
from mxnet import visualization
from graphviz import Digraph
import numpy as np

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='act1', act_type="relu")


fc3 = mx.symbol.FullyConnected(data = act1, name='fc3', num_hidden= 10)
mlp = mx.symbol.Softmax(data = fc3, name = 'mlp')

fc4 = mx.symbol.FullyConnected(data = act1, name='fc4', num_hidden= 10)
mlp1 = mx.symbol.Softmax(data = fc4, name = 'mlp1')

mlp = mx.symbol.Group([mlp,mlp1])
#draw network

batch_size = 100
data_shape = (batch_size, 784)
dot = mx.viz.plot_network(mlp, shape={"data":data_shape})
dot.render('test-output/round-table.gv', view=True)


# data

train_iter, val_iter = mnist_iterator(batch_size=100, input_shape = (784,))

# train by model

logging.basicConfig(level=logging.DEBUG)
'''
model = mx.model.FeedForward(
    ctx = mx.cpu(), symbol = mlp , num_round = 3,
    learning_rate = 0.1, momentum = 0.9, wd = 0.00001)
'''
# model.fit(X=train_iter, eval_data=val_iter)


# ==================train by sinple_bind==============

# build executor
executor = mlp.simple_bind(ctx=mx.cpu(), data=data_shape, grad_req='write')

# get data from executor
arg_arrays = executor.arg_arrays
grad_arrays = executor.grad_arrays
aux_arrays = executor.aux_arrays
output_arrays = executor.outputs
    
args = dict(zip(mlp.list_arguments(), arg_arrays))
grads = dict(zip(mlp.list_arguments(), grad_arrays))
outputs = dict(zip(mlp.list_outputs(), output_arrays))
aux_states = dict(zip(mlp.list_auxiliary_states(), aux_arrays))    

# function for initializing weight and bias

def Init(key, arr):
    if "weight" in key:
        arr[:] = mx.random.uniform(-0.07, 0.07, arr.shape)
        # or
        # arr[:] = np.random.uniform(-0.07, 0.07, arr.shape)
    elif "gamma" in key:
        # for batch norm slope
        arr[:] = 1.0
    elif "bias" in key:
        arr[:] = 0
    elif "beta" in key:
        # for batch norm bias
        arr[:] = 0

# Init args
for key, arr in args.items():
    Init(key, arr)
    
# design a SGD function 

def SGD(key, weight, grad, lr=0.1, grad_norm=batch_size):
    # key is key for weight, we can customize update rule
    norm = 1.0 / grad_norm
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
    else:
        pass
    
#  function to calculate accuracy

def Accuracy(label, pred_prob):
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(label == pred) * 1.0 / label.shape[0]

# training process

num_round = 3
keys = mlp.list_arguments()
# we use extra ndarray to save output of net
pred_prob = mx.nd.zeros(executor.outputs[0].shape)
for i in range(num_round):
    train_iter.reset()
    val_iter.reset()
    train_acc = 0.
    val_acc = 0.
    nbatch = 0.
    # train
    print args.keys()
    for data, label in train_iter:
        # copy data into args   
        args["data"][:] = data # or we can ```data.copyto(args["data"])```
        args["mlp_label"][:] = label
        # multi-label
        #args["mlp1_label"][:] = label
        executor.forward(is_train=True)
        pred_prob[:] = executor.outputs[0]
        executor.backward()
        for key in keys:
            SGD(key, args[key], grads[key])
        train_acc += Accuracy(label.asnumpy(), pred_prob.asnumpy())
        nbatch += 1.
    logging.info("Finish training iteration %d" % i)
    train_acc /= nbatch
    nbatch = 0.
    # eval
    for data, label in val_iter:
        args["data"][:] = data
        executor.forward(is_train=False)
        pred_prob[:] = executor.outputs[0]
        val_acc += Accuracy(label.asnumpy(), pred_prob.asnumpy())
        nbatch += 1.
    val_acc /= nbatch
    logging.info("Train Acc: %.4f" % train_acc)
    logging.info("Val Acc: %.4f" % val_acc)




