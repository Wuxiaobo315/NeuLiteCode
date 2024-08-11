import math
import numpy as np
import torch


def centering(K):
    n = K.shape[0]
    # unit = np.ones([n, n])
    unit = torch.ones([n,n])
    # I = np.eye(n)
    I = torch.eye(n)
    H = I - unit / n

    H = H.to('cuda')

    return torch.mm(torch.mm(H,K),H)
    # return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

import torch
def linear_HSIC(X, Y):
    # L_X = np.dot(X, X.T)
    L_X = torch.mm(X,X.T).to('cuda')
    # L_Y = np.dot(Y, Y.T)
    L_Y = torch.mm(Y,Y.T).to('cuda')

    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    # X = X.detach().cpu()
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__=='__main__':
    np.random.seed(1)
    X = np.random.randn(100, 3*64*64)
    np.random.seed(1000)
    Y = np.random.randn(100, 10)

    z = np.random.randn(100,3*32*32)
    w = np.random.randn(100,64*16*16)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, z)))

    # print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    # print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))
    # print('RBF Kernel CKA, between X and z: {}'.format(linear_HSIC(z, w)))