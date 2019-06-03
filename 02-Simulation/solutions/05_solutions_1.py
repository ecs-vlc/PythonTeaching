#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:05:54 2019

@author: gparkes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
plt.rcParams['figure.figsize'] = 15, 10

# task 1
def task_1():
    N = 500
    dt = 1 / N
    dW = np.sqrt(dt) * np.random.randn(N)
    # set to 0 start
    W = np.cumsum(dW) - dW[0]
    t = np.linspace(0, 1, N)

    plt.plot(t,W,'rx-')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$W$")
    plt.show()


def task_2():
    N = 500
    M1 = 10
    M2 = 10**5
    dt = 1 / N
    t = np.linspace(0, 1, N)
    dW = np.sqrt(dt) * np.random.randn(N,M1)
    dW2 = np.sqrt(dt) * np.random.randn(N,M2)
    W1 = np.cumsum(dW, axis=0) - dW[0,:]
    W2 = np.cumsum(dW2, axis=0) - dW2[0,:]
    for i in range(M1):
        plt.plot(t, W1[:,i])
    plt.plot(t, np.mean(W1,axis=1),'k-')
    plt.plot(t, np.mean(W2,axis=1), 'k-')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$W$")
    plt.show()


# task 3
def euler_maruyama_step(X_n, dt, dW_n, lamda=2., mu=1.):
    fn = lamda * X_n
    gn = mu * X_n
    return X_n + fn*dt + gn*dW_n


# task 4
def euler_maruyama(N, dt, dW, X_0):
    X = np.zeros(N)
    X[0]=X_0
    for n in range(N-1):
        X[n+1] = euler_maruyama_step(X[n], dt, dW[n], 2., 1.)
    return X

def task_4():
    N = 100
    dt = 1/N
    dW = np.sqrt(dt)*np.random.randn(N)
    X = euler_maruyama(N, dt, dW, 1.)


def task_5():
    N = 100

    def f_exact(t, W, lamda, mu, X0):
        return X0 * np.exp((lamda-.5*mu**2)*t + mu*W)

    t, dt = np.linspace(0, 1, N+1, retstep=True)
    dW = np.sqrt(dt)*np.random.randn(N+1)
    W = np.cumsum(dW) - dW[0]
    X = euler_maruyama(len(t), dt, dW, 1.)
    X_exact = f_exact(t, W, 2., 1., 1.)

    plt.plot(t, X_exact, 'k-', label='expected')
    plt.plot(t, X, 'm--', label='realised')
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$X$")
    plt.show()


# task 6