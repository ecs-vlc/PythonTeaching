# Data Analysis in Python

This package of Jupyter notebooks covers the basics of `pandas`, and `networkx` packages, foundational packages for Data Science use. We will be focusing on the loading and preprocessing of datasets of wildly different data types and styles, with a number of end-of-chapter long examples.

Some of the highlights of Pandas include:
- A *fast and efficient* DataFrame object for data manipulation with integrated indexing
- Intelligent data alignment and handling of *missing data*.
- Flexible *reshaping* and pivoting of datasets.
- *Time-series* functionality with date range generation, frequency conversion
- Highly *optimized for performance*, with critical code paths within in Cython or C.

There are questions at the end of each Jupyter notebook to test knowledge, with the complete solutions provided in the `solutions/` folder.

Requirements:

1. `numpy`
4. `matplotlib`
5. Other performance libraries (optional); `cython`, `numba`

Prerequisites:

1. Basic level of Python
2. Introduction to NumPy
3. Introduction to Matplotlib (optional but helpful)

## Pandas Basics

**Pandas** is a core library package, but on top of **NumPy** and extensively used by the Data Science community, where it provides a powerful heterogenous data structure for data analysis. 

In this first section you will cover:
- `pandas.Series` object
- `pandas.DataFrame` object
- `pandas.Index` object
- Data Indexing and Selection
- Indexers
- Operations
- Missing value handling
- Reading and Writing to Files
- Basic aggregation
- Sorting
- Counting
- Complex Strings

For example to create a 1D series of elements we could call:

```python
>>> import pandas as pd
>>> pd.Series([3, 5, 7, 1], index=["Apples","Oranges","Pears","Lemons"])
Apples 	3
Oranges 5
Pears 	7
Lemons 	1
dtype: int64
```

A lot of the functionalities from `numpy` map over to Pandas, except that the index allows for elementwise mapping between Pandas `Series` or `DataFrame` objects.

***

Inspirations: 
1. https://github.com/jakevdp/PythonDataScienceHandbook
2. The [Pandas cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html)
***

The course is covered as interactive Jupyter notebooks which makes things considerably easier. In order to follow this course, we recommend you download Python using the Anaconda distribution (found [here](https://www.anaconda.com/download/)) as this also provides most of the packages used in this course.

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.

All rights reserved.