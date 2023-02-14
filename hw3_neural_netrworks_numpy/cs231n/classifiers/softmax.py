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

    # sanity checking...
    print("X.shape ", X.shape)
    print("y.shape ", y.shape)
    print("W.shape ", W.shape)
    assert X.shape[1] == W.shape[0], "ERROR: W and X need to have same dimension D"
    assert X.shape[0] == y.shape[0], "ERROR: X and Y need to have same dimension C"

    N, D = X.shape
    C = W.shape[1]
    print("N={} D={} C={}".format(N,D,C))

    # compute z=(X*W)
    z = np.zeros([N,C])
    for n in range(N):
      for d in range(D):
        for c in range(C):
           z[n,c] += X[n,d] * W[d,c]

    # compute exp_z = exp(z)
    exp_z = np.zeros([N,C])
    for n in range(N):
      for c in range(C):
        exp_z[n,c] = np.exp(z[n,c])

    # compute sum(exp_z) across each 'row' 
    sum_exp_z = np.zeros([N])
    for n in range(N):
      for c in range(C):
        sum_exp_z[n] += exp_z[n,c]

    # compute the softmax pr_y_predicted
    pr_y_predicted = np.zeros([N,C])
    for n in range(N):
       pr_y_predicted[n:] = exp_z[n,:] / sum(exp_z[n])

    # # zona prints
    # print(z)
    # print(exp_z)
    # print(sum_exp_z)
    # print(pr_y_predicted)

    # now the cross-entropy...
    # ZONA CONFIRM WHICH IS FIRST...
    h_n = np.zeros([N])
    for n in range(N):
      # y (given) is a specific class - that is - probability is 0 or 1
      # it makes no sense to take log, so we conclude that it must be the first term...
      y_given_class = y[n]
      # log(2 = 1) i.e. the first term == 1
      # zona prints...
      # print("--- n={} ---".format(n))
      # print("  y_given_class {}".format(y_given_class))
      # print("  pr_y_predicted {}".format(pr_y_predicted[n,y_given_class]))
      h_n[n] = 1. * np.log(pr_y_predicted[n,y_given_class])

    # average the losses across the batch
    loss = -np.mean(h_n)

    # ZONA
    dW = 0


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

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
