#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:53:19 2019

@author: gparkes
"""
import numpy as np
import matplotlib.pyplot as plt

# task 1
def monte_carlo_integrate(f, dx, dy, N):
    area = (dx[1] - dx[0])*(dy[1] - dy[0])
    # generate random numbers in 2-d
    pairs = np.random.rand(N,2)
    # move pairs into domain [x,y]
    pairs[:,0] *= dx[1] - dx[0]
    pairs[:,0] += dx[0]
    pairs[:,1] *= dy[1] - dy[0]
    pairs[:,1] += dy[0]
    # x is in [:,0]
    integrand = f(pairs[:,0])
    # choose k where random numbers y fall below the integrand
    k = pairs[:,1] < integrand

    return (area * np.sum(k)) / N


# task 2
def task_2():
    def f(x):
        return np.sin(1/(x*(2-x)))**2
    I = monte_carlo_integrate(f, [0, 2], [0, 1], 10**5)
    print(I)


# task 3
def pi(x):
    return np.sqrt(4-x**2)

Nvals = 100*2**np.arange(0,15)
errs = np.zeros((15,))

for i, N in enumerate(Nvals):
    errs[i] = abs(monte_carlo_integrate(pi, [0, 2], [0, 2], N ) - np.pi)


# task 4
def task_4():
    plt.loglog(Nvals, errs, 'kx')
    plt.xlabel(r"$N$")
    plt.ylabel(r"$E$")
    plt.title("Convergence plot of Monte Carlo Integration")
    plt.show()

# task 5
def task_5():
    plt.loglog(Nvals, errs, 'kx')
    m,b = np.polyfit(np.log(Nvals), np.log(errs), 1)
    plt.loglog(Nvals, np.exp(b)*Nvals**m, 'b--')
    plt.show()
