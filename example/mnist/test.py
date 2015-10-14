import mxnet as mx
import numpy as np
import logging
from graphviz import Digraph
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from data import mnist_iterator

# we can use mx.sym in short of mx.symbol
data = mx.sym.Variable("data")
fc1 = mx.sym.FullyConnected(data=data, num_hidden=128, name="fc1")
bn1 = mx.sym.BatchNorm(data=fc1, name="bn1")
act1 = mx.sym.Activation(data=bn1, name="act1", act_type="tanh")
fc2 = mx.sym.FullyConnected(data=act1, name="fc2", num_hidden=10)
softmax = mx.sym.Softmax(data=fc2, name="softmax")
# visualize the network
batch_size = 100
data_shape = (batch_size, 784)
dot = mx.viz.plot_network(softmax, shape={"data":data_shape})
#dot.render('test-output/round-table.gv', view=True)


# context different to ```mx.model```, 
# In mx.model, we wrapped parameter server, but for a single executor, the context is only able to be ONE device
# run on cpu
ctx = mx.cpu()
# run on gpu
# ctx = mx.gpu()
# run on third gpu
# ctx = mx.gpu(2)
executor = softmax.simple_bind(ctx=ctx, data=data_shape, grad_req='write')
# The default ctx is CPU, data's shape is required and ```simple_bind``` will try to infer all other required 
# For MLP, the ```grad_req``` is write to, and for RNN it is different



# get argument arrays
arg_arrays = executor.arg_arrays
# get grad arrays
grad_arrays = executor.grad_arrays
# get aux_states arrays. Note: currently only BatchNorm symbol has auxiliary states, which is moving_mean and moving_var
aux_arrays = executor.aux_arrays
# get outputs from executor
output_arrays = executor.outputs

args = dict(zip(softmax.list_arguments(), arg_arrays))
grads = dict(zip(softmax.list_arguments(), grad_arrays))
outputs = dict(zip(softmax.list_outputs(), output_arrays))
aux_states = dict(zip(softmax.list_auxiliary_states(), aux_arrays))

# we can print the args we have
print("args: ", args)
print("-" * 20)
print("grads: ", grads)
print("-" * 20)
print("aux_states: ", aux_states)
print("-" * 20)
print("outputs: ", outputs)


# helper function
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


def SGD(key, weight, grad, lr=0.1, grad_norm=batch_size):
    # key is key for weight, we can customize update rule
    # weight is weight array
    # grad is grad array
    # lr is learning rate
    # grad_norm is scalar to norm gradient, usually it is batch_size
    norm = 1.0 / grad_norm
    # here we can bias' learning rate 2 times larger than weight
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
    else:
        pass
    
    
train, val = mnist_iterator(batch_size=100, input_shape = (784,))
train_iter = train
val_iter = val

def Accuracy(label, pred_prob):
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(label == pred) * 1.0 / label.shape[0]


num_round = 3
keys = softmax.list_arguments()
print keys
# we use extra ndarray to save output of net
pred_prob = mx.nd.zeros(executor.outputs[0].shape)
for i in range(num_round):
    train_iter.reset()
    val_iter.reset()
    train_acc = 0.
    val_acc = 0.
    nbatch = 0.
    # train
    for data, label in train_iter:
        # copy data into args
        args["data"][:] = data # or we can ```data.copyto(args["data"])```
        args["softmax_label"][:] = label
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