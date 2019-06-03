#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:48:37 2019

@author: gparkes
"""
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# task 1
def life_step_dask(x):

    def access_roll(x):
        #da.roll extracts an array where all values are shifted left, right, up or down
        l_roll = da.roll(x,1,axis=0)
        r_roll = da.roll(x,-1,axis=0)
        return l_roll + r_roll + da.roll(x,1,axis=1) + da.roll(x,-1,axis=1) + \
            da.roll(l_roll, 1, axis=1) + da.roll(l_roll, -1, axis=1) + \
            da.roll(r_roll, 1, axis=1) + da.roll(r_roll, -1, axis=1)

    c_grid = da.map_overlap(x, access_roll, depth=1, boundary="periodic")

    nx = x - ((x == 1) & ((c_grid < 2) | (c_grid > 3))).astype(np.int)
    nx = nx + ((x == 0) & (c_grid == 3)).astype(np.int)
    return nx

N=10
x = da.random.randint(2, size=(N,N), chunks=(5,5))
y = life_step_dask(x).compute()
print(np.array(x))
print(y)


# task 2
def task_2():
    N=10000
    chunksize = 1000
    x = da.random.randint(2, size=(N,N), chunks=(chunksize,chunksize))
    y = life_step_dask(x).compute()
    print(np.array(x))
    print(y)


def task_3():
    N=100
    chunksize = 50
    steps = 16

    x = da.random.randint(2, size=(N,N), chunks=(chunksize,chunksize))

    fig,ax=plt.subplots(ncols=4, nrows=4, figsize=(16,8))

    current = x
    for i in range(steps):
        curr_arr = life_step_dask(current).compute()
        # re-create dask array from numpy.array
        current = da.from_array(curr_arr, chunks=(chunksize,chunksize))
        # plot
        ax[i%4,int(i/4)].imshow(curr_arr)
        ax[i%4,int(i/4)].set_title(i)


def task_4():
    N=1000
    chunksize = 500
    steps = 200

    x = da.random.randint(2, size=(N,N), chunks=(chunksize,chunksize))
    t = np.arange(steps)
    s_tot = np.zeros(steps)

    fig,ax=plt.subplots()

    current = x
    for i in range(steps):
        curr_arr = life_step_dask(current).compute()
        # compute sum
        s_tot[i] = curr_arr.mean()
        # re-create dask array from numpy.array
        current = da.from_array(curr_arr, chunks=(chunksize,chunksize))

    ax.plot(t, s_tot, 'kx-')
    ax.set_xlabel(r"steps")
    ax.set_ylabel("proportion of units alive")
    plt.show()