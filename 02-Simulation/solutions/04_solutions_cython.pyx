#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:01:53 2019

@author: gparkes
"""

%%cython

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef extern from "time.h":
    long int time(int)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calc_positional_e(np.ndarray[np.int_t, ndim=2] spins, int i, int j, double J):
    """
    In this strategy, we use pure Python and beef up our code with JIT or Cython.
    """
    cdef:
        int N
        double spin_l, spin_r, spin_u, spin_d

    N = spins.shape[0]

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

    return spin_l + spin_r + spin_u + spin_d


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double delta_e(np.ndarray[np.int_t, ndim=2] spins, int i, int j, double J):
    return 2.0 * J * spins[i,j] * calc_positional_e(spins, i, j, J)


cdef double total_mag_e(np.ndarray[np.int_t, ndim=2] spins, double J):
    cdef:
        int i, j, N
        double sum_sp

    N = spins.shape[0]
    sum_sp = 0
    for i in range(N):
        for j in range(N):
            sum_sp += (calc_positional_e(spins, i, j, J))

    return -J*sum_sp


@cython.boundscheck(False)
@cython.wraparound(False)
def metropolis_hastings2(int N, int n_steps, double beta):
    # declare
    cdef:
        int s, randx, randy
        double J, dE
        np.ndarray[np.float_t, ndim=1] E_series = np.ones(n_steps)
        np.ndarray[np.int_t, ndim=2] spins = np.random.choice([-1,1], size=(N,N))

    srand48(123456789)
    J = 1.0
    E_series[0] = total_mag_e(spins, J)

    for s in range(1, n_steps):
        randx = int(drand48() * N)
        randy = int(drand48() * N)
        # calculate de
        dE = delta_e(spins, randx, randy, J)
        # if we decrease the energy or meet boltzmann requirements, flip the spin
        if dE < 0. or exp(-beta*dE) > drand48():
            spins[randx, randy] *= -1
            E_series[s] = E_series[s-1] + dE
        else:
            E_series[s] = E_series[s-1]
    return spins, E_series