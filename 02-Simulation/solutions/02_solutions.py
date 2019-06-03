#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:40:20 2019

@author: gparkes
"""
import numpy as np
from scipy import integrate, ndimage
import matplotlib.pyplot as plt

def task_1():
    x = 10. / 100. # x is a constant because we are dealing with a cubical container.
    a = 0. # plate rests at the surface of the water
    w = 9800. # density of water
    b = 10. / 100. # plate goes 10cm into water bowl

    # function to integrate
    def f(y, x):
        return x*y

    F_int = integrate.quad(f, a, b, args=(x))
    F = w * F_int[0]
    print("Applying %.4fN force to plate." % F)


def task_2():
    m = 5.
    c = 10.
    k = 128.
    t = np.linspace(0, 10, 100)
    # initial conditions
    x0 = [0., 0.6]
    # convert to system of first-order differential equations
    # function to diff.

    # we set w(t)=x'(t) so:
    # w'(t)=(-cx'(t) - kx(t)) / m
    def pend(y, t, m, c, k):
        x, omega = y
        dydt = [omega, (-c*omega - k*x) / m]
        return dydt

    # create solution
    sol = integrate.odeint(pend, x0, t, args=(m,c,k))

    plt.plot(t, sol[:,0], label=r"$x(t)$")
    plt.plot(t, sol[:,1], label=r"$\omega(t)$")
    plt.xlabel(r"$t$")
    plt.legend()
    plt.show()

A = plt.imread("bigcat.jpg")

def task_3():
    fig,axes=plt.subplots(ncols=2, figsize=(10,4))
    axes[0].imshow(A)
    axes[1].imshow(ndimage.laplace(A))
    plt.show()


def task_4():
    sigs = np.logspace(-1,0.7,9)
    fig,ax = plt.subplots(ncols=3, nrows=3, figsize=(15,10))
    for i,s in enumerate(sigs):
        ax[i%3,int(i/3)].imshow(ndimage.laplace(ndimage.gaussian_filter(A, s)))
        ax[i%3,int(i/3)].set_title(r"$\sigma$=%.3f" % s)


def task_5():
    fig,ax = plt.subplots(ncols=3, figsize=(16,4))
    a_set, _ = ndimage.label(A > A.mean())
    ax[0].imshow(a_set)
    b_mat = ndimage.laplace(ndimage.gaussian_filter(A, 2.2))
    b_set, _ = ndimage.label(b_mat > b_mat.mean())
    ax[1].imshow(b_set)
    c_mat = ndimage.sobel(ndimage.laplace(ndimage.gaussian_filter(A, 1.8)))
    c_set, _ = ndimage.label(c_mat > c_mat.mean())
    ax[2].imshow(c_set)
    plt.show()
