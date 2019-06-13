# Numerical Simulation in Python

This package of Jupyter notebooks covers the basics of `numpy`, `scipy` and `dask` packages, which are the foundation of many other exciting packages in Python - and we provide some in-depth simulation studies to gain a strong grasp of usage for these libraries.

There are questions at the end of each Jupyter notebook to test knowledge, with the complete solutions provided in the `solutions/` folder.

Requirements:

1. `numpy`
2. `scipy`
3. `dask`
4. `matplotlib`
5. Other performance libraries (optional); `cython`, `numba`

Prerequisites:

1. Basic level of Python
2. `itertools`

## Basics to NumPy

**NumPy** is a core library package for scientific computing in Python, whereby it provides a high-performance multi-dimensional array/matrix object, written in C with a Python header, and a host of functions for array creation, access, manipulation and deletion. For example arrays are created as:

```python
>>> import numpy as np
>>> np.array([1., 2., 3.])
array([1., 2., 3.])
```

By default, NumPy arrays are float64-bit. We can create random numbers using the following:

```python
>>> import numpy as np
>>> np.random.rand(5, 2)
array([[.664, .234, .113, .764, .987],
	   [.096, .338, .654, .622, .491]])
```

One of the most powerful aspects of NumPy is to perform vector or matrix operations across all the values in an array with one function, rather than needing to iterate over every element in a vector:

```python
>>> import numpy as np
>>> x = np.zeros(5, 2)
>>> y = np.random.rand(5,2)
>>> print(x - y)
array([[-0.012, -0.455, -0.365, -0.776, -0.199],
	   [-0.542, -0.611, -0.848, -0.209, -0.674]])
```

Or aggregate vectors or matrices together:

```python
>>> import numpy as np
>>> x = np.random.normal(0, 1., size=(10000,))
>>> print(np.mean(x))
0.0016746
```

There is this, and much more within this comprehensive basics introduction.

## Basics to SciPy

With **NumPy** acting as the bread-and-butter for scientific computing, SciPy expands this by building on top of NumPy to provide a vast selection of functions that operate on `numpy` arrays, mainly used for scientific and engineering applications. SciPy is a huge ecosystem of packages, of which here are some of the more used:

| Subpackage | Description |
| ----------- | ---------------------------------- | 
| `cluster` | clustering algorithms |
| `integrate` | Integration and ordinary differential equation solvers |
| `optimize` | Optimization and root-finding algorithms |
| `sparse` | Provides sparse matrices and data structures |
| `stats` | Statistical distributions and functions |
| `linalg` | Linear algebra functions |

It is recommended that only the packages you need from SciPy are imported, given that the ecosystem is very large. For example if we just wanted Linear Algebra we could:

```python
>>> from scipy import linalg
>>> b = linalg.solve(A, x)
```

To get a solid grasp of this package, you'll have to work through the notebook!

## Dask arrays

The `dask` framework builds on top of NumPy and/or Pandas (a later library) to essentially provide flexible parallel computing for analytics. `dask` functions are similar to `itertools` as they are declarative and produce a task graph which is then executed when `.compute()` is called:

```python
>>> import dask.array as da
>>> x = da.random.randn((100000,100000), chunks=(1000,1000)).mean()
>>> print(x.compute())
0.0000145366
```

here we specify a `chunks` parameter whereby `dask` breaks this large array into the chunksize specified and computes them individually, then aggregating the results together.

A primary advantage of this approach is that the memory required for using `dask` is proportional to the chunk size, not the size of the total problem. Hence parallelizable problems suddenly become incredibly easy and fast to perform, irrespective of the total size needed. There are many mechanisms for handling programs that need to work on the *boundary* of the chunks, or need to incorporate some form of communication between blocks, but this obviously decreases performance time and scalability. 

`dask` effortlessly also allows users to set up their data structures to an online or local server/cluster. This is all managed in the background, with no additional bulk-code needed. It also scales-down incredibly well, as `dask` is just as effective when running on a laptop with a single-threaded process, enabling large-scale flexibility to the computational resources available at a given time to the user.

## Simulations

We provide 4 large case-study examples (2 in each notebook) for users to tackle in their own time, which aim to cover a large number of scientific and numerical disciplines. Use NumPy, SciPy, Dask and other performance tools at your leisure to tackle these problems, solutions are provided!

1. Monte Carlo Integration
2. Ising Model
3. Stochastic Differential Equations
4. Molecular Dynamics

There are more exciting simulations to try within the `Extras/` package at the front of the LearnPython course.

***

Inspirations: 
1. https://github.com/ageron/handson-ml
2. https://github.com/jakevdp/PythonDataScienceHandbook

***

The course is covered as interactive Jupyter notebooks which makes things considerably easier. In order to follow this course, we recommend you download Python using the Anaconda distribution (found [here](https://www.anaconda.com/download/)) as this also provides most of the packages used in this course.

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.

All rights reserved.