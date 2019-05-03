# Python Course

Welcome to this Python Course.

The aim of this repository is to provide complex examples to the Python programming language and extended libraries. It contains many *multidisciplinary* examples to write code for, on a range of coding topic areas, particularly for research purposes.

## Principles

In principles we aim to cover the basics of the Python programming language and more - in this module we draw only from internal modules/packages that are in-built into the language. We start from saying 'Hello World'; a classic introduction to any programming language:

```python
print("Hello World")
```

All the way through to more performance-friendly implementations using the powerful `itertools` internal module, and performing complex string searching and extraction using *Regex*. This module is definitely for those either with or without previous programming experience, and will throw you in the deep end with complex tasks at the end of each notebook. 

## Simulation

Simulation is now considered to be the *third pillar of science*, alongside observation and experimentation. Gaining skills to develop simulations that closely resemble real-world applications is highly desirable that not only contributes to high-paying jobs, but also to push forward the frontiers of engineering and science. 

With simulation, users will be introduced to the extremely powerful `NumPy` package with it's C-style performance array and matrix operations. 

```python
import numpy as np
x = np.array([0., 1., 1., 3.])
```

In addition, users will learn plenty of mathematical operations to be performed on computer such as differentiation, integration, and interpolation. We introduce `SciPy` for scientific computing, as well as performance-related practice using `Dask` to enable scalability for large problems that can be automatically distributed to supercomputer clusters. 

```python
import dask.array as da
x = da.arange(25, chunks=5).compute()
```

Users will enjoy multi-disciplinary experience in two notebooks with complex tasks that involve building simulations taking hours to complete. 

## Data

In the internet age, there is now more data than ever accessible and the list of problem domains that intelligently use data is growing exponentially. Gaining skills to preprocess and manipulate data sensibly are in high demand. 

This module will introduce you to `Pandas`, a flexible, high-performance data structure for handling relational data, which is built on top of `NumPy`. 

```python
import pandas as pd
x = pd.Series(data=[3, 1, 4], index=["apples", "oranges", "bananas"])
```

This includes complex database manipulations including merges, unions and intersects, aggregation and reshaping. 

| | Value | Patient | Phylum |
 | --- | ------- | ----- |:------------------:|
| **0** | 632 | 1 | Firmicutes |
| **1** | 1638 | 1 | Proteobacteria |
| **2** | 433 | 2 | Firmicutes |
| **3** | 1130 | 2 | Proteobacteria |

We also explore the use of graph data using *nodes* and *edges* with the powerful `NetworkX` package. Users will be introduced to a number of complex tasks and examples where they manipulate datasets from multiple disciplines.

## Visualisation

With large amounts of data, this presents challenges as to how to articulate results in an effective manner, knowledge of powerful visualisation tools and methods will enable users to present research or findings effectively.

We introduce a number of different packages, most of which are built on `Matplotlib`, ported from *MATLAB*, and include `Seaborn` and `Plotly`. 

![Unable to display image](https://github.com/gregparkes/PythonCourse/04-Visualization/sample_img.png)

Users will become very familiar with basic plots, and will progress onto animated and interactive plots to model complex problems that change in space and time. 

***

Inspirations: 
1. https://github.com/ageron/handson-ml
2. https://github.com/jakevdp/PythonDataScienceHandbook

***

The course is covered as interactive Jupyter notebooks which makes things considerably easier. In order to follow this course, we recommend you download Python using the Anaconda distribution (found [here](https://www.anaconda.com/download/)) as this also provides most of the packages used in this course.

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.