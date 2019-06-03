#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:47:03 2019

@author: gparkes
"""
import numpy as np
import matplotlib.pyplot as plt

# task 1
def lennard_jones_potential(r):
    return 24. * (2. * (1 / r)**14 - (1 / r)**8)


# task 2
def acceleration(x, L, Rc):
    a = np.zeros_like(x)
    n, p = x.shape
    # loop
    for i in range(n):
        dx = x[i, :] - x
        s = np.abs(dx) > L/2
        dx[s] -= np.sign(dx[s])*L
        for j in range(i+1, n):
            r_ij = np.sqrt(np.dot(dx[j, :], dx[j, :]))
            if r_ij < Rc:
                phi_r = lennard_jones_potential(r_ij)
                a[i, :] += dx[j, :]*phi_r
                a[j, :] -= dx[j, :]*phi_r
    return a


# task 3
def verlet(x, v, a, dt, L):
    x = x + dt*v + .5*dt**2 * a
    # boundary check
    x[x < 0] += L
    x[x > L] -= L
    vstar = v + .5*dt*a
    a = acceleration(x, L, 2.5)
    v = vstar + .5*dt*a
    return x, v, a


# task 4
def task_4():
    x = np.array([[4., 0., 0.],[4.+2.**(1/12),0.,0.]])
    L = 10
    Rc= 2.5
    dt = 0.1
    steps = 500
    v = np.zeros_like(x)
    a = acceleration(x, L, Rc)

    fig, ax = plt.subplots()
    pos = np.zeros((2, 3, steps))

    for i in range(steps):
        x, v, a = verlet(x, v, a, dt, L)
        pos[:, :, i] = x.copy()

    ax.scatter(pos[0, 0, :], pos[0, 1, :])
    ax.scatter(pos[1, 0, :], pos[1, 1, :])
    plt.show()


# task 5
def calc_temperature(v, L):
    E = np.zeros((len(v)))
    # loop
    for particle in range(len(v)):
        E[particle] = .5*(L**2)*(np.linalg.norm(v[particle]))**2
    return np.sum(E) * (2. / (3. * len(E)))


def task_5():
    steps = 500
    x = np.array([[4., 0., 0.],[4.+2.**(1/12),0.,0.]])
    v = np.zeros_like(x)
    Rc= 2.5
    L = 6.1984
    a = acceleration(x, L, Rc)
    dt = 0.005
    L = 6.1984
    T = np.zeros(steps)

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i in range(steps):
        x, v, a = verlet(x, v, a, dt, L)
        T[i] = calc_temperature(v, L)
        ax2.scatter(x[:,0], x[:,1])

    ax.plot(np.arange(0,steps*dt,dt), T, 'k-')
    ax.set_xlabel("time t")
    ax.set_ylabel("Temperature T")
    plt.show()
