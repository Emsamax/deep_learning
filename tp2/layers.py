from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.stats as stats

from im2col_cython import col2im_cython, im2col_cython

zero_init = np.zeros

def variance_scaling_initializer(shape, fan_in, factor=2.0, seed=None):
  sigma = np.sqrt(factor / fan_in)
  return stats.truncnorm(-2, 2, loc=0, scale=sigma).rvs(shape)


# -- ABSTRACT CLASS DEFINITION --
class Layer(metaclass = ABCMeta):
  "Interface for layers"
  # See documentation of abstract base classes (ABC): https://docs.python.org/3/library/abc.html

  @abstractmethod
  def forward(self, inputs):
    """
    Args:
      inputs: ndarray tensor.
    Returns:
      ndarray tensor, result of the forward pass.
    """
    pass

  @abstractmethod
  def backward_inputs(self, grads):
    """
    Args:
      grads: gradient of the loss with respect to the output of the layer.
    Returns:
      Gradient of the loss with respect to the input of the layer.
    """
    pass

  def backward_params(self, grads):
    """
    Args:
      grads: gradient of the loss with respect to the output of the layer.
    Returns:
      Gradient of the loss with respect to all the parameters of the layer as a list
      [[w0, g0], ..., [wk, gk], self.name] where w are parameter weights and g their gradient.
      Note that wk and gk must have the same shape.
    """
    pass


# -- CONVOLUTION LAYER --
class Convolution(Layer):
  "N-dimensional convolution layer"

  def __init__(self, input_layer, num_filters, kernel_size, name, padding='SAME',
               weights_initializer_fn=variance_scaling_initializer,
               bias_initializer_fn=zero_init):
    self.input_shape = input_layer.shape
    N, C, H, W = input_layer.shape
    self.C = C
    self.N = N
    self.num_filters = num_filters
    self.kernel_size = kernel_size

    assert kernel_size % 2 == 1

    self.padding = padding
    if padding == 'SAME':
      # with zero padding
      self.shape = (N, num_filters, H, W)
      self.pad = (kernel_size - 1) // 2
    else:
      # without padding
      self.shape = (N, num_filters, H - kernel_size + 1, W - kernel_size + 1)
      self.pad = 0

    fan_in = C * kernel_size**2
    self.weights = weights_initializer_fn([num_filters, kernel_size**2 * C], fan_in)
    self.bias = bias_initializer_fn([num_filters])
    # this implementation doesn't support strided convolutions
    self.stride = 1
    self.name = name
    self.has_params = True

  def forward(self, x):
    k = self.kernel_size
    self.x_cols = im2col_cython(x, k, k, self.pad, self.stride)
    res = self.weights.dot(self.x_cols) + self.bias.reshape(-1, 1)
    N, C, H, W = x.shape
    out = res.reshape(self.num_filters, self.shape[2], self.shape[3], N)
    return out.transpose(3, 0, 1, 2)

  def backward_inputs(self, grad_out):
    # nice trick from CS231n, backward pass can be done with just matrix mul and col2im
    grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
    grad_x_cols = self.weights.T.dot(grad_out)
    N, C, H, W = self.input_shape
    k = self.kernel_size
    grad_x = col2im_cython(grad_x_cols, N, C, H, W, k, k, self.pad, self.stride)
    return grad_x

  def backward_params(self, grad_out):
    grad_bias = np.sum(grad_out, axis=(0, 2, 3))
    grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
    grad_weights = grad_out.dot(self.x_cols.T).reshape(self.weights.shape)
    return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class MaxPooling(Layer):
  def __init__(self, input_layer, name, pool_size=2, stride=2):
    self.name = name
    self.input_shape = input_layer.shape
    N, C, H, W = self.input_shape
    self.stride = stride
    self.shape = (N, C, H // stride, W // stride)
    self.pool_size = pool_size
    assert pool_size == stride, 'Invalid pooling params'
    assert H % pool_size == 0
    assert W % pool_size == 0
    self.has_params = False

  def forward(self, x):
    N, C, H, W = x.shape
    self.input_shape = x.shape
    # with this clever reshaping we can implement pooling where pool_size == stride
    self.x = x.reshape(N, C, H // self.pool_size, self.pool_size,
                       W // self.pool_size, self.pool_size)
    self.out = self.x.max(axis=3).max(axis=4)
    # if you are returning class member be sure to return a copy
    return self.out.copy()

  def backward_inputs(self, grad_out):
    grad_x = np.zeros_like(self.x)
    out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (self.x == out_newaxis)
    dout_newaxis = grad_out[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, grad_x)
    # this is almost the same as the real backward pass
    grad_x[mask] = dout_broadcast[mask]
    # in the very rare case that more then one input have the same max value
    # we can aprox the real gradient routing by evenly distributing across multiple inputs
    # but in almost all cases this sum will be 1
    grad_x /= np.sum(mask, axis=(3, 5), keepdims=True)
    grad_x = grad_x.reshape(self.input_shape)
    return grad_x


class Flatten(Layer):
  def __init__(self, input_layer, name):
    self.input_shape = input_layer.shape
    self.N = self.input_shape[0]
    self.num_outputs = 1
    for i in range(1, len(self.input_shape)):
      self.num_outputs *= self.input_shape[i]
    self.shape = (self.N, self.num_outputs)
    self.has_params = False
    self.name = name

  def forward(self, inputs):
    self.input_shape = inputs.shape
    inputs_flat = inputs.reshape(self.input_shape[0], -1)
    self.shape = inputs_flat.shape
    return inputs_flat

  def backward_inputs(self, grads):
    return grads.reshape(self.input_shape)


class FC(Layer):
  def __init__(self, input_layer, num_outputs, name,
               weights_initializer_fn=variance_scaling_initializer,
               bias_initializer_fn=zero_init):

    self.input_shape = input_layer.shape
    self.N = self.input_shape[0]
    self.shape = (self.N, num_outputs)
    self.num_outputs = num_outputs

    self.num_inputs = 1
    # (C*H*W) entry size
    for i in range(1, len(self.input_shape)):
      self.num_inputs *= self.input_shape[i]

    self.weights = weights_initializer_fn([self.num_inputs, num_outputs], fan_in=self.num_inputs)
    self.bias = bias_initializer_fn([num_outputs])
    self.name = name
    self.has_params = True

  def forward(self, inputs):
    # Reshape x if it's not  2D (flatten C*H*W into one dimension)
    # stock X_flattened for backward
    self.inputs = inputs.reshape(self.N, self.num_inputs)
    
    # Forward pass: Y = X @ W + B
    # X (N, D) @ W (D, M) -> Z (N, M)
    out = np.dot(self.inputs, self.weights) + self.bias
    
    # return out 
    return out

  def backward_inputs(self, grads):
    # chain derivative dL/dX = dL/dZ @ dZ/dX
    # dZ/dX is W.T (transpose of W)
    # dL/dX (N, D) = dL/dZ (N, M) @ W.T (M, D)
    grad_x = np.dot(grads, self.weights.T)
    # If original input was 4D, we must reshape grad_x back to its original shape
    grad_x = grad_x.reshape(self.input_shape)
    return grad_x

  def backward_params(self, grads):
    """
    Args:
      grads: ndarray of shape (N, num_outputs) (dL/dZ)
    Returns:
      List of params and gradient pairs.
    """
    
    # dL/dW = X.T @ dL/dZ
    # X.T (D, N) @ dL/dZ (N, M) -> dL/dW (D, M)
    # dL/dW has the same shape as W: (num_inputs, num_outputs)
    # self.inputs is already X_flattened (N, num_inputs)
    grad_weights = np.dot(self.inputs.T, grads)
    
    # dL/dB = sum(dL/dZ, axis=0) (sum over the batch dimension N)
    # grad_bias has shape (num_outputs,)
    grad_bias = np.sum(grads, axis=0)
    
    # return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]
    return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class ReLU(Layer):
  """ 
  activation layer compute f(x) = max(0, x)
  """
  def __init__(self, input_layer, name):
    self.shape = input_layer.shape
    self.name = name
    self.has_params = False

  def forward(self, inputs):
    #stock x for backward
    self.inputs = inputs
    #out is activation function f(x) = max(0, x)
    out = np.maximum(0, inputs)
    #return out 
    return out 

  def backward_inputs(self, grads):
    #dout/dX
    #derivate of relu is f'(x) = x if x > 0  else 0
    mask = (self.inputs > 0)
    # chain derivative : dL/dx = (dL/dout) * (dout/dx)
    grad_x = grads * mask
    return grad_x


class SoftmaxCrossEntropyWithLogits():
  def __init__(self):
    self.has_params = False
    # Placeholder for softmax output (probabilities) needed for backward pass
    self.probs = None 

  def forward(self, x, y):
    # stock x logitsfor backward 
    self.x = x 
    # compute average loss over N examples. (stable)
    x_max = np.max(x, axis=1, keepdims=True)
    
    # Compute the exponentials
    exp_x = np.exp(x - x_max)
    
    # Compute the sum of exponentials for normalization
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    # Compute the probabilities and stock the probabilities for backward
    self.probs = exp_x / sum_exp_x
    
    #Cross-Entropy Loss (L = -sum(y * log(p)))
    # Clip probabilities to prevent log(0) which is -inf.
    probs_clipped = np.clip(self.probs, 1e-12, 1.0)
    
    # y is one-hot = -log(p) for the true class.
    sample_losses = -np.sum(y * np.log(probs_clipped), axis=1)
    
    #Compute the average loss over the batch N
    N = x.shape[0]
    average_loss = np.sum(sample_losses) / N
    
    # return average loss
    return average_loss

  def backward_inputs(self, x, y):
    # compute loss gradient to x
    
    # The gradient dL/dx for Softmax + Cross-Entropy is simply (P - Y)
    # dL/dx = P - Y
    grad_x = self.probs - y
    
    # We must divide the gradient by the batch size N
    N = x.shape[0]
    grad_x /= N
    return grad_x

class L2Regularizer():
  def __init__(self, weights, weight_decay, name):
    # this is still a reference to original tensor so don't change self.weights
    self.weights = weights
    self.weight_decay = weight_decay
    self.name = name

  def forward(self):
    # L2 Norm Squared: ||W||^2 = sum(W^2)
    l2_norm_sq = np.sum(self.weights ** 2)
    
    # Calculate the L2 regularization loss: L_reg = 0.5 * lambda * ||W||^2
    l2_loss = 0.5 * self.weight_decay * l2_norm_sq
    
    # return the scalar loss
    return l2_loss

  def backward_params(self):
    # dL_reg/dW = lambda * W
    grad_weights = self.weight_decay * self.weights
    
    # return a list containing the weight tensor and its gradient, plus the name
    # The format is [[parameter_tensor, gradient_tensor], layer_name]
    return [[self.weights, grad_weights], self.name]
  
class RegularizedLoss():
  def __init__(self, data_loss, regularizer_losses):
    self.data_loss = data_loss
    self.regularizer_losses = regularizer_losses
    self.has_params = True
    self.name = 'RegularizedLoss'

  def forward(self, x, y):
    loss_val = self.data_loss.forward(x, y)
    for loss in self.regularizer_losses:
      loss_val += loss.forward()
    return loss_val

  def backward_inputs(self, x, y):
    return self.data_loss.backward_inputs(x, y)

  def backward_params(self):
    grads = []
    for loss in self.regularizer_losses:
      grads += [loss.backward_params()]
    return grads

