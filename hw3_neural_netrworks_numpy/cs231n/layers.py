from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    flattened_x = x.reshape((N,-1))

    out = np.dot(flattened_x, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    shape_x = x.shape
    shape_w = w.shape

    N = shape_x[0]
    flattened_x = x.reshape((N,-1))

    dx_flat = np.dot(dout, w.T)
    dx = np.reshape(dx_flat, shape_x)

    dw_flat = np.dot(flattened_x.T, dout)
    dw = np.reshape(dw_flat, shape_w)

    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.where(x>0., x, 0.)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * np.where(x>0., 1., 0.)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # print("zona x.shape\n",x.shape)
    # print("zona y.shape\n",y.shape)
    
    N = x.shape[0]
    C = x.shape[1]

    # get a 1-hot version of y
    # n_values = np.max(y) + 1  # ZONA BUG
    n_values = C
    y_one_hot = np.eye(n_values)[y]

    # forward:

    # numerically stabilize before exp()
    # x_max = x.max(axis=1).reshape(-1,1)
    # x = x - x_max

    exp_x = np.exp(x)
    sum_by_row = np.sum(exp_x, axis=1)
    sum_by_row_tiled = np.tile(sum_by_row,(C,1)).T
    softmax = np.exp(x)/sum_by_row_tiled
    # cross entropy is sum( pr(y) * log(pr(y_predicted)) )
    cross_entropy_samples = np.sum( y_one_hot * np.log(softmax), axis=1)
    loss = -np.mean(cross_entropy_samples)

    # gradient
    # reference https://deepnotes.io/softmax-crossentropy
    m = y.shape[0]
    grad = softmax
    grad[range(m),y] -= 1
    grad = grad/m
    dx = grad
    # return grad

    # # backward:
    # # reference https://e2eml.school/softmax.html

    # grad_w = np.zeros((N,C))
    # for n in range(N):
    #   d_softmax = np.ones((C,C)) * 999.  # initialize to some bogus number
    #   # print("softmax\n",softmax)
    #   # print("softmax.shape",softmax.shape)
    #   # print("softmax[0]\n",softmax[0])
    #   sts = softmax[n]  # softmax this sample
    #   # print("sts\n",sts)
    #   d_softmax = (                                                           
    #       sts * np.identity(sts.size)                                 
    #       - sts.transpose() @ sts)

    #   # print("y_one_hot.shape",y_one_hot.shape)
    #   # downstream_grad = y_one_hot[n]  # y this sample
    #   downstream_grad = np.ones(C)
    #   # input_grad = downstream_grad @ d_softmax
    #   # print("downstream_grad",downstream_grad)
    #   # print("d_softmax",d_softmax)
    #   input_grad = np.dot(downstream_grad, d_softmax)
    #   print("input_grad", input_grad)
    #   grad_w[n] = input_grad
    #   # assert False, "hoooo"

    # # print("grad_w\n",grad_w)
    # # print("grad_w.shape",grad_w.shape)
    # # assert False, "hold upp"
    # # dx = np.dot(x.T, grad_w.T)
    # dx = grad_w
    # # print("grad_w\n",grad_w)
    # # print("x.shape ", x.shape)
    # # print("grad_w.shape", grad_w.shape)
    # # print("dx.shape ", dx.shape)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
