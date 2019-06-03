#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:53:19 2019

@author: gparkes
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt

# task 1
@jit(nopython=True)
def calc_positional_e(spins, J):
    """
    In this strategy, we use pure Python and beef up our code with JIT or Cython.
    """
    N = spins.shape[0]
    E = np.zeros_like(spins)
    # for loop
    for i in range(N):
        for j in range(N):
            if i == 0:
                spin_l = spins[N-1,j]
                spin_r = spins[i+1,j]
            elif i == N-1:
                spin_l = spins[i-1,j]
                spin_r = spins[0,j]
            else:
                spin_l = spins[i,j-1]
                spin_r = spins[i,j+1]

            if j == 0:
                spin_u = spins[i,N-1]
                spin_d = spins[i,j+1]
            elif j == N-1:
                spin_u = spins[i,j-1]
                spin_d = spins[i,0]
            else:
                spin_u = spins[i,j-1]
                spin_d = spins[i,j+1]

            E[i,j] = spin_l + spin_r + spin_u + spin_d
    return E


# task 2
@jit
def delta_e(spins, i, j, J):
    N = spins.shape[0]
    # check boundaries first!
    if i == 0:
        spin_l = spins[N-1,j]
        spin_r = spins[i+1,j]
    elif i == N-1:
        spin_l = spins[i-1,j]
        spin_r = spins[0,j]
    else:
        spin_l = spins[i-1,j]
        spin_r = spins[i+1,j]

    if j == 0:
        spin_u = spins[i,N-1]
        spin_d = spins[i,j+1]
    elif j == N-1:
        spin_u = spins[i,j-1]
        spin_d = spins[i,0]
    else:
        spin_u = spins[i,j-1]
        spin_d = spins[i,j+1]

    return 2. * J * spins[i,j] * (spin_l+spin_r+spin_u+spin_d)


# task 3
def total_mag_e(spins, J):
    return -J*np.sum(calc_positional_e(spins, J))


# task 4
def metropolis_hastings(N, n_steps, beta):
    spins = np.random.choice([-1,1], size=(N,N))
    J = 1
    E_series = np.ones(n_steps)
    E_series[0] = total_mag_e(spins, J)

    for s in range(1, n_steps):
        randx = np.random.randint(N)
        randy = np.random.randint(N)
        # calculate de
        dE = delta_e(spins, randx, randy, J)
        # if we decrease the energy or meet boltzmann requirements, flip the spin
        if dE < 0. or np.exp(-beta*dE) > np.random.rand():
            spins[randx, randy] *= -1
            E_series[s] = E_series[s-1] + dE
        else:
            E_series[s] = E_series[s-1]
    return spins, E_series


# task 5
def task_5():
    A = metropolis_hastings(40, int(5*10**5), 0.1)
    B = metropolis_hastings(40, int(5*10**5), 1.0)

    fig,ax=plt.subplots(ncols=3, figsize=(16,4))

    ax[0].plot(A[1], label=r"$\beta=0.1$")
    ax[0].plot(B[1], label=r"$\beta=1.0$")
    ax[0].legend()
    ax[1].imshow(A[0], cmap="gray")
    ax[2].imshow(B[0], cmap='gray')

    for a in [ax[1], ax[2]]:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    ax[0].set_xlabel("steps")
    ax[0].set_ylabel(r"Energy $E$")
    plt.show()


# task 6
def task_6():
    bvals = 50
    betas = np.linspace(0.1, 0.6, bvals)
    m_ser = np.zeros(bvals)

    def average_magnetization(spins, N):
        return (1 / N**2) * np.sum(spins)

    for b in range(bvals):
        # print("Running beta=%.4f" % betas[b])
        C, E = metropolis_hastings(20, 500000, betas[b])
        m_ser[b] = average_magnetization(C[0], 20)

    # plot
    fig = plt.figure(figsize=(13,5))
    plt.plot(betas, m_ser, "g*")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$M$")
    plt.show()


# task 7

"""
Uncomment this code below to run in Jupyter notebook
"""
# %prun metropolis_hastings(40, 500000, 0.5)

# %timeit metropolis_hastings(40, 500000, 0.5)

# after running cython block...

# %timeit metropolis_hastings2(40, 500000, 0.5)

def task_7():
    C1, E1 = metropolis_hastings(40, 500000, .6)
    C2, E2 = metropolis_hastings2(40, 500000, .6)
    C3, E3 = metropolis_hastings(40, 500000, .1)
    C4, E4 = metropolis_hastings2(40, 500000, .1)
    fig,ax=plt.subplots(ncols=2, figsize=(16,4))

    ax[0].plot(E1, label=r"python")
    ax[0].plot(E2, label="cython")
    ax[0].legend()
    ax[1].plot(E3, label=r"python")
    ax[1].plot(E4, label="cython")
    ax[1].legend()
