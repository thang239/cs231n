import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  this convolutional network will contain following architecture:
  [conv-relu-conv-relu-pool]xN - [affine-relu]xM - affine-[softmax or SVM]

  """
  
  def __init__(self, input_dim=(3, 32, 32), M=3, N=2, num_filters=None, filter_size=3,
               hidden_dims=None,use_batchnorm=False, num_classes=10, weight_scale=1e-3, reg=0.0,dropout=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    - M: number of combined convolutional layers
    - N: number of affine layers
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.num_layers = 2 * M + N + 1
    self.M = M
    self.N = N
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    (C, H, W) = input_dim
    mu,sigma = 0,weight_scale
    HH = filter_size
    WW = filter_size
    #Initialize hidden dims for affine layers
    if hidden_dims==None:
        hidden_dims = (np.arange(N)+1)*200
    if num_filters==None:
        num_filters = 2**(np.arange(M).repeat(2))*32
    #Initialize first N combine layers
    last = C
    for i in range(1,M*2+1):
        num_filter = num_filters[i-1]

        self.params['W'+str(i)] = sigma * np.random.randn(num_filter, last, HH, WW)
        self.params['b'+str(i)] = np.zeros(num_filter)
        # print self.params['W'+str(i)].shape
        if self.use_batchnorm:
            self.params['gamma'+str(i)] = np.ones(num_filter)
            self.params['beta'+str(i)] = np.zeros(num_filter)
        last = num_filter
    # Weight of output of M combine layers is H * W * last divide by 4^(number of pool(2x2))
    D = (H * W * last)/(4**M)
    for j in range(N):
        hidden_dim = hidden_dims[j]
        i+=1
        self.params['W'+str(i)] = sigma * np.random.randn(D,hidden_dim)
        self.params['b'+str(i)] = np.zeros(hidden_dim)
        if self.use_batchnorm:
            self.params['gamma'+str(i)] = np.ones(hidden_dim)
            self.params['beta'+str(i)] = np.zeros(hidden_dim)
        D = hidden_dim
    #Initialize last affine layer
    i+=1
    self.params['W'+str(i)] = sigma * np.random.randn(D,num_classes)
    self.params['b'+str(i)] = np.zeros(num_classes)

    # for k,v in self.params.iteritems():
    #     if "W" in k:
    #         print k
    #         print self.params[k].shape
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      # if seed is not None:
      #   self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # W1, b1 = self.params['W1'], self.params['b1']
    # W2, b2 = self.params['W2'], self.params['b2']
    # W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.params['W1'].shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = []
    in_data = X
    #Forward M combine layers
    for i in range(1,self.M*2+1):
        W = self.params['W'+str(i)]  
        b = self.params['b'+str(i)]
        g = self.params['gamma'+str(i)]
        beta = self.params['beta'+str(i)]
        if i%2==1:
            if self.use_batchnorm:
                out,t_cache = conv_batch_relu_forward(in_data, W, b, conv_param, self.bn_params[i-1], g, beta)
            else:
                out, t_cache = conv_relu_forward(in_data, W, b, conv_param)
            cache.append(t_cache)
        else:
            if self.use_batchnorm:
                out, t_cache = conv_batch_relu_pool_forward(in_data, W, b, conv_param, self.bn_params[i-1], g, beta, pool_param)
            else:
                out, t_cache = conv_relu_pool_forward(in_data, W, b, conv_param, pool_param)
            cache.append(t_cache)
        in_data = out

    # Forward N affine layers

    for i in range(1,self.N+1):
        W = self.params['W'+str(2*self.M+i)]
        b = self.params['b'+str(2*self.M+i)]
        g = self.params['gamma'+str(2*self.M+i)]
        beta = self.params['beta'+str(2*self.M+i)]
        if self.use_batchnorm:
            out,t_cache = affine_batchnorm_relu_forward(out, W, b, self.bn_params[2*self.M+i-1],g, beta)
        else:
            out, t_cache = affine_relu_forward(out, W, b)
        cache.append(t_cache)
        if self.use_dropout:
            out,t_cache = dropout_forward(out, self.dropout_param)
            cache.append(t_cache)
    #Forward last  affine layers
    W = self.params['W'+str(self.num_layers)]
    b = self.params['b'+str(self.num_layers)]
    scores, af_cache = affine_forward(out, W, b)
    cache.append(af_cache)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    for i in xrange(self.num_layers):
      loss += 0.5 * self.reg * np.sum(self.params['W' + `i + 1`] ** 2)
    NN = self.num_layers
    index = NN
    #Backprop last layer
    t_cache = cache.pop()
    dout, grads['W'+str(index)], grads['b'+str(index)] = affine_backward(dout,t_cache)

    #Backprop N affine layer
    for i in range(self.N):
        index-=1
        t_cache = cache.pop()
        if self.use_dropout:
            dout = dropout_backward(dout, t_cache)
            t_cache = cache.pop()
        if self.use_batchnorm:
            dout, grads['W'+str(index)], grads['b'+str(index)], grads['gamma'+str(index)], grads['beta'+str(index)] = affine_batchnorm_relu_backward(dout,t_cache)    
        else:
            dout, grads['W'+str(index)], grads['b'+str(index)] = affine_relu_backward(dout,t_cache)
    #Backprop M combined conv layers
    for i in range(self.M*2):
        t_cache = cache.pop()
        index-=1

        if i%2==0:
            if self.use_batchnorm:
                dout, grads['W'+str(index)], grads['b'+str(index)], grads['gamma'+str(index)], grads['beta'+str(index)] = conv_batch_relu_pool_backward(dout, t_cache)    
            else:
                dout, grads['W'+str(index)], grads['b'+str(index)] = conv_relu_pool_backward(dout,t_cache)
        else:
            if self.use_batchnorm:
                dout, grads['W'+str(index)], grads['b'+str(index)], grads['gamma'+str(index)], grads['beta'+str(index)] = conv_batch_relu_backward(dout, t_cache)
            else:
                dout, grads['W'+str(index)], grads['b'+str(index)] = conv_relu_backward(dout,t_cache)

    # for i in xrange(3):
    #   loss += 0.5 * self.reg * np.sum(self.params['W' + `i + 1`] ** 2)

    # dout, grads['W3'], grads['b3'] = affine_backward(dout, cache_af)

    # dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache_ar)

    # dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache_crp)

    for k,v in grads.iteritems():
        if "W" in k:
            grads[k] += self.reg * self.params[k]






    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
