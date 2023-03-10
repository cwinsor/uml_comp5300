from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # --- SOFTMAX LOSS (description...) ---
    #
    # given X (NxD), W (DxC)...
    # 
    # first step is to compute softmax from X, W.
    # For a particular z_i this is:
    # softmax(z_i) = exp(z_i) / sum(exp(z_c)) for all c in C
    # 
    # a) compute z = X dot W        (NxD) * (D*C) ==> (NxC)
    # b) numerator = exp_z = exp(z)     (NxC)
    # c) denominator = sum_exp_z by "row" (Nx1)
    # d) softmax = pr_y_predicted = numerator / denominator by "row"  (NxC)
    # each row is a random variable (a set of probabilities that sum to 1.0)
    # 
    # second step is the loss and for softmax this is cross-entropy
    # we will use "h" for cross-entropy
    # in concept if we are given two vectors (a, b) of same length C then
    # h = sum_over_vector( pr(a) * log(pr(b)) )
    # We have pr_y_predicted but y is not in that form - it is 1-hot
    # BUT note since y only has one element as "1" the others are "0" we can
    # skip creating the 1-hot vectors and instead just use the number that
    # is in "y" (it's the only one that will contribute).

    N, D = X.shape
    C = W.shape[1]

    # compute z=(X*W)
    z = np.zeros([N,C])
    for n in range(N):
      for d in range(D):
        for c in range(C):
           z[n,c] += X[n,d] * W[d,c]

    # numerically stabilize before taking exp()
    z_max = z.max(axis=1).reshape(-1,1)
    z = z - z_max
    exp_z = np.exp(z)

    # compute exp_z = exp(z)
    exp_z = np.zeros([N,C])
    for n in range(N):
      for c in range(C):
        exp_z[n,c] = np.exp(z[n,c])

    # exp_z goes two places. I am making two "copies" so that
    # I can keep straight when doing the backprop.
    exp_z_numerator = exp_z
    exp_z_denominator = exp_z

    # denominator - compute sum(exp_z) across each 'row' 
    exp_z_denom_sum = np.zeros([N])
    for n in range(N):
      for c in range(C):
        exp_z_denom_sum[n] += exp_z_denominator[n,c]

    exp_z_denom_sum_bcast = np.tile(exp_z_denom_sum,(C,1)).T

    # compute the softmax pr_y_predicted
    pr_y_predicted = np.zeros([N,C])
    for n in range(N):
      for c in range(C):
        pr_y_predicted[n,c] = exp_z_numerator[n,c] / exp_z_denom_sum_bcast[n,c]

    # now the cross-entropy...
    # h = sum( p(y) * log(p(y_predicted)) )
    # there is no need to run the sum here since the
    # the first term y is the true/given class and is 1-hot.
    # It identifies exactly one class with probability = 1 and others 0.
    h_n = np.zeros([N])
    for n in range(N):
      y_given_class = y[n]
      h_n[n] = 1. * np.log(pr_y_predicted[n,y_given_class])

    # average the losses across the batch
    loss = -np.mean(h_n)

    # --------- GRADIENT (attempt 2) -------
    n_values = C
    y_one_hot = np.eye(n_values)[y]

    # reference https://e2eml.school/softmax.html
    grad_w = np.zeros((N,C))
    for n in range(N):
      softmax = y_one_hot[n]
      softmax = np.reshape(softmax, (1, -1))
      grad = np.ones(C)
      d_softmax = (                                                           
          softmax * np.identity(softmax.size)                                 
          - softmax.transpose() @ softmax)
      input_grad = grad @ d_softmax
      grad_w[n] = input_grad

    dW = np.dot(X.T, grad_w)

    # # --------- GRADIENT (attempt 1) -------
    # # we are working backward from the output of the "softmax" part of the network...
    # # we use same naming scheme but prefix with "grad_"
    # grad_out = 1.

    # # grad backward through the divide:  out = exp_z / sum_exp_z
    # # grad for division f(x,y) == x/y == x * y^-1
    # # df_dx = 1/y
    # # df_dy = -x/y^-2
    # # numerator:
    # grad_exp_z_numerator = grad_out / exp_z_denom_sum_bcast  # should be NxC
    
    # # denominator:
    # grad_exp_z_denom_sum_bcast = grad_out * -1. * exp_z_numerator / exp_z_denom_sum_bcast / exp_z_denom_sum_bcast

    # # grad backward through the broadcast in the denominator
    # # the broadcast occurs because the sum reduces it to a column vector, then that is
    # # applied to all classes. Broadcast (branch) is an "add" in backprop
    # grad_at_denom_branch = np.sum(grad_exp_z_denom_sum_bcast, axis=1)

    # # grad backward through the "sum" in the denominator
    # # backprop of gradient through "+" (i.e. "sum") duplicates the gradient to each of the contributing terms
    # # reference https://www.youtube.com/watch?v=d14TUNcbn1k&t=138s at 34:14 minutes
    # # there are "C" terms
    # grad_exp_z_denominator = np.tile(grad_at_denom_branch,(C,1)).T

    # # grad backward through the "branch" that occurs when z goes two places (numerator, denominator)
    # # rule for branch is the source gradient is the sum of the destination gradients
    # grad_exp_z = grad_exp_z_numerator + grad_exp_z_denominator

    # # grad backward through the exp_z = exp(z)
    # # grad here is exp(z)
    # grad_z = grad_exp_z  # should be NxC

    # # grad backwards through  z = X dot W
    # #  dz/dw = W.T
    # #  dz/dx = X.T
    # # grad_x = np.dot(grad_z, W.T)  # should be NxD (not used anyway)
    # grad_w = np.dot(X.T, grad_z)  # should be DxC... <----what we want

    # dW = grad_w

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = X.shape
    C = W.shape[1]

    z = np.dot(X,W)

    # numerically stabilize before exp()
    z_max = z.max(axis=1).reshape(-1,1)
    z = z - z_max
    exp_z = np.exp(z)

    # numerator
    numerator = exp_z

    # denominator
    sum_exp_z = np.sum(exp_z, axis=1)
    denominator = np.tile(sum_exp_z,(C,1)).T

    pr_y_predicted = numerator / denominator

    # the cross entropy...
    # convert y to 1-hot...
    n_values = C
    y_one_hot = np.eye(n_values)[y]

    # cross entropy is sum( pr(y) * log(pr(y_predicted)) )
    cross_entropy_samples = np.sum( y_one_hot * np.log(pr_y_predicted), axis=1)
    loss = -np.mean(cross_entropy_samples)


    # --------- GRADIENT -------
    # reference https://e2eml.school/softmax.html
    grad_w = np.zeros((N,C))
    for n in range(N):
      softmax = y_one_hot[n]
      grad = np.ones(C)
      d_softmax = (                                                           
          softmax * np.identity(softmax.size)                                 
          - softmax.transpose() @ softmax)
      input_grad = grad @ d_softmax
      grad_w[n] = input_grad
    dW = np.dot(X.T, grad_w)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
