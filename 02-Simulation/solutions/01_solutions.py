#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:34:24 2019

@author: gparkes
"""

import numpy as np
import matplotlib.pyplot as plt

def task_1():
    N = 1000
    p = 20
    _ = plt.hist(np.sum(np.random.rand(p,N),axis=0)-np.sum(np.random.rand(p,N),axis=0))


# task 2
def fisher_wright(P, s, mu, nu, Tmax):
    t = 0
    n = np.zeros((Tmax+1), dtype=np.int64)
    while n[t]<P and t<Tmax:
        # select
        p_s = (1+s)*n[t] / (P+s*n[t])
        # mutate
        p_sm = (1-nu)*p_s + mu*(1.-p_s)
        # sample
        t += 1
        n[t] = np.random.binomial(P, p_sm)
    return n[:t+1]


def task_3():
    param_sets = [
        (200, 0.1, 0.001, 0.001, int(10**3)),
        (200, .1, .1, .001, int(10**3)),
        (200, 0, .001, .001, int(10**3)),
        (200, .1, .001, .1, int(10**3)),
    ]
    names = [r"$P=200, s=10^{-1}, \mu=10^{-3}, \nu=10^{-3}$",
         r"$P=200, s=10^{-1}, \mu=10^{-1}, \nu=10^{-3}$",
         r"$P=200, s=0, \mu=10^{-3}, \nu=10^{-3}$",
         r"$P=200, s=10^{-1}, \mu=10^{-3}, \nu=10^{-1}$"
    ]
    fig, axes = plt.subplots(figsize=(8,6))
    for p, name in zip(param_sets, names):
        nt = fisher_wright(*p)
        axes.plot(np.arange(nt.shape[0]), nt, label=name)
    axes.set_xlabel(r"$t$")
    axes.set_ylabel(r"$n$ mutants")
    axes.legend(loc="upper right")
    plt.show()


# task 4
def fisher_wright_modified(P, s, mu, nu, Tmax, Nr):
    t = 0
    n = np.zeros((Tmax+1, Nr), dtype=np.int32)
    # once a certain proportion (say 10%) of the matrix contains total population P, we stop iterating.
    prop = ((Tmax+1)*Nr) * 0.1
    while np.sum(n==P) < prop and t < Tmax:
        p_s = (1+s)*n[t,:] / (P+s*n[t,:])
        # mutate
        p_sm = (1-nu)*p_s + mu*(1.-p_s)
        # sample
        t += 1
        n[t,:] = np.random.binomial(P, p_sm, size=(Nr,))
    return n[:t+1,:]


def task_4():
    nt2 = fisher_wright_modified(200, 0.1, 0.001, 0.001, int(10**3), int(10**4))
    ntm = nt2.mean(axis=1)
    ntsd = nt2.std(axis=1)
    t = np.arange(len(nt2))

    plt.plot(t, ntm, 'k-')
    plt.fill_between(t, ntm - 2*ntsd, ntm + 2*ntsd, color='r', alpha=.4)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$n \,$ mutants")

    plt.show()
