#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:36:42 2019

@author: gparkes

This script contains demonstration code for all sorts of sci-kit learn examples
all functions generate a graph, by default we do not save these to a file

Material drawn from:
    ttps://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb

Taken and used for non-commercial purposes, with modification.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances_argmin
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_blobs, make_swiss_roll

import misc_fig as mf

def demo_broadcasting_cubes():

    #------------------------------------------------------------
    # Draw a figure and axis with no boundary
    fig = plt.figure(figsize=(6, 4.5), facecolor='w')
    ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)

    # types of plots
    solid = dict(c="black", ls="-", lw=1, label_kwargs=dict(color="k"))
    dotted = dict(c='black', ls='-', lw=0.5, alpha=0.5, label_kwargs=dict(color='gray'))
    depth = .3

    #------------------------------------------------------------
    # Draw top operation: vector plus scalar
    mf.draw_cube(ax, (1, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    mf.draw_cube(ax, (2, 10), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
    mf.draw_cube(ax, (3, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)

    mf.draw_cube(ax, (6, 10), 1, depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '5', **solid)
    mf.draw_cube(ax, (7, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)
    mf.draw_cube(ax, (8, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)

    mf.draw_cube(ax, (12, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '5', **solid)
    mf.draw_cube(ax, (13, 10), 1, depth, [1, 2, 3, 6, 9], '6', **solid)
    mf.draw_cube(ax, (14, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '7', **solid)

    ax.text(5, 10.5, '+', size=12, ha='center', va='center')
    ax.text(10.5, 10.5, '=', size=12, ha='center', va='center')
    ax.text(1, 11.5, r'${\tt np.arange(3) + 5}$',
            size=12, ha='left', va='bottom')

    #------------------------------------------------------------
    # Draw middle operation: matrix plus vector

    # first block
    mf.draw_cube(ax, (1, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)
    mf.draw_cube(ax, (2, 7.5), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
    mf.draw_cube(ax, (3, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '1', **solid)

    mf.draw_cube(ax, (1, 6.5), 1, depth, [2, 3, 4], '1', **solid)
    mf.draw_cube(ax, (2, 6.5), 1, depth, [2, 3], '1', **solid)
    mf.draw_cube(ax, (3, 6.5), 1, depth, [2, 3, 7, 10], '1', **solid)

    mf.draw_cube(ax, (1, 5.5), 1, depth, [2, 3, 4], '1', **solid)
    mf.draw_cube(ax, (2, 5.5), 1, depth, [2, 3], '1', **solid)
    mf.draw_cube(ax, (3, 5.5), 1, depth, [2, 3, 7, 10], '1', **solid)

    # second block
    mf.draw_cube(ax, (6, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    mf.draw_cube(ax, (7, 7.5), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
    mf.draw_cube(ax, (8, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)

    mf.draw_cube(ax, (6, 6.5), 1, depth, range(2, 13), '0', **dotted)
    mf.draw_cube(ax, (7, 6.5), 1, depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (8, 6.5), 1, depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)

    mf.draw_cube(ax, (6, 5.5), 1, depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)
    mf.draw_cube(ax, (7, 5.5), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (8, 5.5), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)

    # third block
    mf.draw_cube(ax, (12, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)
    mf.draw_cube(ax, (13, 7.5), 1, depth, [1, 2, 3, 6, 9], '2', **solid)
    mf.draw_cube(ax, (14, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '3', **solid)

    mf.draw_cube(ax, (12, 6.5), 1, depth, [2, 3, 4], '1', **solid)
    mf.draw_cube(ax, (13, 6.5), 1, depth, [2, 3], '2', **solid)
    mf.draw_cube(ax, (14, 6.5), 1, depth, [2, 3, 7, 10], '3', **solid)

    mf.draw_cube(ax, (12, 5.5), 1, depth, [2, 3, 4], '1', **solid)
    mf.draw_cube(ax, (13, 5.5), 1, depth, [2, 3], '2', **solid)
    mf.draw_cube(ax, (14, 5.5), 1, depth, [2, 3, 7, 10], '3', **solid)

    ax.text(5, 7.0, '+', size=12, ha='center', va='center')
    ax.text(10.5, 7.0, '=', size=12, ha='center', va='center')
    ax.text(1, 9.0, r'${\tt np.ones((3,\, 3)) + np.arange(3)}$',
            size=12, ha='left', va='bottom')

    #------------------------------------------------------------
    # Draw bottom operation: vector plus vector, double broadcast

    # first block
    mf.draw_cube(ax, (1, 3), 1, depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '0', **solid)
    mf.draw_cube(ax, (1, 2), 1, depth, [2, 3, 4, 7, 10], '1', **solid)
    mf.draw_cube(ax, (1, 1), 1, depth, [2, 3, 4, 7, 10], '2', **solid)

    mf.draw_cube(ax, (2, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)
    mf.draw_cube(ax, (2, 2), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (2, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)

    mf.draw_cube(ax, (3, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)
    mf.draw_cube(ax, (3, 2), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (3, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)

    # second block
    mf.draw_cube(ax, (6, 3), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    mf.draw_cube(ax, (7, 3), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
    mf.draw_cube(ax, (8, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)

    mf.draw_cube(ax, (6, 2), 1, depth, range(2, 13), '0', **dotted)
    mf.draw_cube(ax, (7, 2), 1, depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (8, 2), 1, depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)

    mf.draw_cube(ax, (6, 1), 1, depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)
    mf.draw_cube(ax, (7, 1), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)
    mf.draw_cube(ax, (8, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)

    # third block
    mf.draw_cube(ax, (12, 3), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    mf.draw_cube(ax, (13, 3), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
    mf.draw_cube(ax, (14, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)

    mf.draw_cube(ax, (12, 2), 1, depth, [2, 3, 4], '1', **solid)
    mf.draw_cube(ax, (13, 2), 1, depth, [2, 3], '2', **solid)
    mf.draw_cube(ax, (14, 2), 1, depth, [2, 3, 7, 10], '3', **solid)

    mf.draw_cube(ax, (12, 1), 1, depth, [2, 3, 4], '2', **solid)
    mf.draw_cube(ax, (13, 1), 1, depth, [2, 3], '3', **solid)
    mf.draw_cube(ax, (14, 1), 1, depth, [2, 3, 7, 10], '4', **solid)

    ax.text(5, 2.5, '+', size=12, ha='center', va='center')
    ax.text(10.5, 2.5, '=', size=12, ha='center', va='center')
    ax.text(1, 4.5, r'${\tt np.arange(3).reshape((3,\, 1)) + np.arange(3)}$',
            ha='left', size=12, va='bottom')

    ax.set_xlim(0, 16)
    ax.set_ylim(0.5, 12.5)

    # save
    # fig.savefig('figures/02.05-broadcasting.png')


def demo_aggregate_df():
    df = pd.DataFrame({'data': [1, 2, 3, 4, 5, 6]},
                       index=['A', 'B', 'C', 'A', 'B', 'C'])
    df.index.name = 'key'


    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = plt.axes([0, 0, 1, 1])

    ax.axis('off')

    mf.draw_dataframe(df, [0, 0])

    for y, ind in zip([3, 1, -1], 'ABC'):
        split = df[df.index == ind]
        mf.draw_dataframe(split, [2, y])

        sum = pd.DataFrame(split.sum()).T
        sum.index = [ind]
        sum.index.name = 'key'
        sum.columns = ['data']
        mf.draw_dataframe(sum, [4, y + 0.25])

    result = df.groupby(df.index).sum()
    mf.draw_dataframe(result, [6, 0.75])

    style = dict(fontsize=14, ha='center', weight='bold')
    plt.text(0.5, 3.6, "Input", **style)
    plt.text(2.5, 4.6, "Split", **style)
    plt.text(4.5, 4.35, "Apply (sum)", **style)
    plt.text(6.5, 2.85, "Combine", **style)

    arrowprops = dict(facecolor='black', width=1, headwidth=6)
    plt.annotate('', (1.8, 3.6), (1.2, 2.8), arrowprops=arrowprops)
    plt.annotate('', (1.8, 1.75), (1.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (1.8, -0.1), (1.2, 0.7), arrowprops=arrowprops)

    plt.annotate('', (3.8, 3.8), (3.2, 3.8), arrowprops=arrowprops)
    plt.annotate('', (3.8, 1.75), (3.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (3.8, -0.3), (3.2, -0.3), arrowprops=arrowprops)

    plt.annotate('', (5.8, 2.8), (5.2, 3.6), arrowprops=arrowprops)
    plt.annotate('', (5.8, 1.75), (5.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (5.8, 0.7), (5.2, -0.1), arrowprops=arrowprops)

    plt.axis('equal')
    plt.ylim(-1.5, 5);

    # fig.savefig('figures/03.08-split-apply-combine.png')


def demo_features_and_labels_grid():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.axis('equal')

    # Draw features matrix
    ax.vlines(range(6), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=0, xmax=5, lw=1)
    font_prop = dict(size=12, family='monospace')
    ax.text(-1, -1, "Feature Matrix ($X$)", size=14)
    ax.text(0.1, -0.3, r'n_features $\longrightarrow$', **font_prop)
    ax.text(-0.1, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    # Draw labels vector
    ax.vlines(range(8, 10), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=8, xmax=9, lw=1)
    ax.text(7, -1, "Target Vector ($y$)", size=14)
    ax.text(7.9, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    ax.set_ylim(10, -2)

    # fig.savefig('figures/05.02-samples-features.png')


def demo_5_fold_cross_validation():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    mf.draw_rects(5, ax, textprop=dict(size=10))

    # fig.savefig('figures/05.03-5-fold-CV.png')


def demo_classification_data(n=50, n_centers=2, std=.6):
    X, y = make_blobs(n_samples=n, centers=n_centers, random_state=0, cluster_std=std)
    # fit
    clf = SVC(kernel="linear").fit(X, y)
    # new points to predict
    X2,_ = make_blobs(n_samples=80, centers=2, random_state=0, cluster_std=.8)
    X2 = X2[50:]
    y2 = clf.predict(X2)

    return X, y, X2, y2, clf


def demo_regression_data(n=200, n2=100):
    rng = np.random.RandomState(1)
    X = rng.randn(n, 2)
    y = np.dot(X, [-2, 1]) + .1 * rng.randn(X.shape[0])

    # fit model
    model = LinearRegression().fit(X, y)
    # create points to predict
    X2 = rng.randn(n2,2)
    y2 = model.predict(X2)

    return X, y, X2, y2, model


def demo_cluster_data(n=100, centers=4, std=1.5):
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=42)
    # fit
    model = KMeans(centers, random_state=0).fit(X)
    y_hat = model.predict(X)

    return X, y, y_hat, model


def demo_swiss_roll_data(n=200, noise=.5):
    X, y = make_swiss_roll(n_samples=n, noise=noise, random_state=42)
    X = X[:, [0, 2]]
    return X, y


def demo_nonlinear_data(n=30, err=.8, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(n,1)**2
    y = 10 - 1. / (X.ravel() + .1)
    if err > 0:
        y += err * rng.randn(n)
    return X, y


def demo_classification_example_1():

    X, y, X2, y2, clf = demo_classification_data()
    # plot the data
    fig, ax = plt.subplots(figsize=(8, 6))
    point_style = dict(cmap='Paired', s=50)
    ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)

    # format plot
    mf.format_plot(ax, 'Input Data')
    ax.axis([-1, 4, -2, 7])
    # fig.savefig('figures/05.01-classification-1.png')


def demo_classification_example_2():

    X, y, X2, y2, clf = demo_classification_data()

    # Get contours describing the model
    xx = np.linspace(-1, 4, 10)
    yy = np.linspace(-2, 7, 10)
    xy1, xy2 = np.meshgrid(xx, yy)
    point_style = dict(cmap='Paired', s=50)
    Z = np.array([clf.decision_function([t])
                  for t in zip(xy1.flat, xy2.flat)]).reshape(xy1.shape)

    # plot points and model
    fig, ax = plt.subplots(figsize=(8, 6))
    line_style = dict(levels = [-1.0, 0.0, 1.0],
                      linestyles = ['dashed', 'solid', 'dashed'],
                      colors = 'gray', linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)
    ax.contour(xy1, xy2, Z, **line_style)

    # format plot
    mf.format_plot(ax, 'Model Learned from Input Data')
    ax.axis([-1, 4, -2, 7])

    # fig.savefig('figures/05.01-classification-2.png')


def demo_regression_example_1():
    X, y, _, _, _ = demo_regression_data()
    # plot data points
    fig, ax = plt.subplots()
    points = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                        cmap='viridis')

    # format plot
    mf.format_plot(ax, 'Input Data')
    ax.axis([-4, 4, -3, 3])

    # fig.savefig('figures/05.01-regression-1.png')


def demo_regression_example_2():

    X, y, _, _, _ = demo_regression_data()

    points = np.hstack([X, y[:, None]]).reshape(-1, 1, 3)
    segments = np.hstack([points, points])
    segments[:, 0, 2] = -8

    # plot points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, s=35,
               cmap='viridis')
    ax.add_collection3d(Line3DCollection(segments, colors='gray', alpha=0.2))
    ax.scatter(X[:, 0], X[:, 1], -8 + np.zeros(X.shape[0]), c=y, s=10,
               cmap='viridis')

    # format plot
    ax.patch.set_facecolor('white')
    ax.view_init(elev=20, azim=-70)
    ax.set_zlim3d(-8, 8)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.set(xlabel='feature 1', ylabel='feature 2', zlabel='label')

    # Hide axes (is there a better way?)
    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)
    for tick in ax.w_xaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_yaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_zaxis.get_ticklines():
        tick.set_visible(False)

    # fig.savefig('figures/05.01-regression-2.png')


def demo_regression_example_3():
    X, y, _, _, model = demo_regression_data()
    # plot data points
    fig, ax = plt.subplots()
    pts = ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                     cmap='viridis', zorder=2)

    # compute and plot model color mesh
    xx, yy = np.meshgrid(np.linspace(-4, 4),
                         np.linspace(-3, 3))
    Xfit = np.vstack([xx.ravel(), yy.ravel()]).T
    yfit = model.predict(Xfit)
    zz = yfit.reshape(xx.shape)
    ax.pcolorfast([-4, 4], [-3, 3], zz, alpha=0.5,
                  cmap='viridis', norm=pts.norm, zorder=1)

    # format plot
    mf.format_plot(ax, 'Input Data with Linear Fit')
    ax.axis([-4, 4, -3, 3])

    # fig.savefig('figures/05.01-regression-3.png')


def demo_cluster_example_1():
    X,_,_,_ = demo_cluster_data()
    # plot the input data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], s=50, color='gray')

    # format the plot
    mf.format_plot(ax, 'Input Data')

    # fig.savefig('figures/05.01-clustering-1.png')


def demo_cluster_example_2():

    X,y,_,_ = demo_cluster_data()
    # plot the data with cluster labels
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap='viridis')

    # format the plot
    mf.format_plot(ax, 'Learned Cluster Labels')

    # fig.savefig('figures/05.01-clustering-2.png')


def demo_dimensionality_reduction_example_1():
    X, _ = demo_swiss_roll_data()
    # visualize data
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color='gray', s=30)

    # format the plot
    mf.format_plot(ax, 'Input Data')

    # fig.savefig('figures/05.01-dimesionality-1.png')


def demo_dimensionality_reduction_example_2():
    X, y = demo_swiss_roll_data()
    # create model
    model = Isomap(n_neighbors=8, n_components=1)
    y_fit = model.fit_transform(X).ravel()

    # visualize data
    fig, ax = plt.subplots()
    pts = ax.scatter(X[:, 0], X[:, 1], c=y_fit, cmap='viridis', s=30)
    cb = fig.colorbar(pts, ax=ax)

    # format the plot
    mf.format_plot(ax, 'Learned Latent Parameter')
    cb.set_ticks([])
    cb.set_label('Latent Variable', color='gray')

    # fig.savefig('figures/05.01-dimesionality-2.png')


def demo_bias_variance_tradeoff():
    X, y = demo_nonlinear_data()
    xfit = np.linspace(-0.1, 1.0, 1000)[:, None]

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    model1 = PolynomialRegression(1).fit(X, y)
    model20 = PolynomialRegression(20).fit(X, y)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    ax[0].scatter(X.ravel(), y, s=40)
    ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
    ax[0].axis([-0.1, 1.0, -2, 14])
    ax[0].set_title('High-bias model: Underfits the data', size=14)

    ax[1].scatter(X.ravel(), y, s=40)
    ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')
    ax[1].axis([-0.1, 1.0, -2, 14])
    ax[1].set_title('High-variance model: Overfits the data', size=14)

    # fig.savefig('figures/05.03-bias-variance.png')


def demo_bias_variance_tradeoff_metrics():
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    X, y = demo_nonlinear_data()
    X2, y2 = demo_nonlinear_data(10, rseed=42)
    xfit = np.linspace(-0.1, 1.0, 1000)[:, None]
    PolynomialRegression = lambda degree: make_pipeline(PolynomialFeatures(degree),\
                                                        LinearRegression())

    model1 = PolynomialRegression(1).fit(X, y)
    model20 = PolynomialRegression(20).fit(X, y)

    ax[0].scatter(X.ravel(), y, s=40, c='blue')
    ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
    ax[0].axis([-0.1, 1.0, -2, 14])
    ax[0].set_title('High-bias model: Underfits the data', size=14)
    ax[0].scatter(X2.ravel(), y2, s=40, c='red')
    ax[0].text(0.02, 0.98, "training score: $R^2$ = {0:.2f}".format(model1.score(X, y)),
               ha='left', va='top', transform=ax[0].transAxes, size=14, color='blue')
    ax[0].text(0.02, 0.91, "validation score: $R^2$ = {0:.2f}".format(model1.score(X2, y2)),
               ha='left', va='top', transform=ax[0].transAxes, size=14, color='red')

    ax[1].scatter(X.ravel(), y, s=40, c='blue')
    ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')
    ax[1].axis([-0.1, 1.0, -2, 14])
    ax[1].set_title('High-variance model: Overfits the data', size=14)
    ax[1].scatter(X2.ravel(), y2, s=40, c='red')
    ax[1].text(0.02, 0.98, "training score: $R^2$ = {0:.2g}".format(model20.score(X, y)),
               ha='left', va='top', transform=ax[1].transAxes, size=14, color='blue')
    ax[1].text(0.02, 0.91, "validation score: $R^2$ = {0:.2g}".format(model20.score(X2, y2)),
               ha='left', va='top', transform=ax[1].transAxes, size=14, color='red')

    # fig.savefig('figures/05.03-bias-variance-2.png')


def demo_validation_curve():
    x = np.linspace(0, 1, 1000)
    y1 = -(x - 0.5) ** 2
    y2 = y1 - 0.33 + np.exp(x - 1)

    fig, ax = plt.subplots()
    ax.plot(x, y2, lw=10, alpha=0.5, color='blue')
    ax.plot(x, y1, lw=10, alpha=0.5, color='red')

    ax.text(0.15, 0.2, "training score", rotation=45, size=16, color='blue')
    ax.text(0.2, -0.05, "validation score", rotation=20, size=16, color='red')

    ax.text(0.02, 0.1, r'$\longleftarrow$ High Bias', size=18, rotation=90, va='center')
    ax.text(0.98, 0.1, r'$\longleftarrow$ High Variance $\longrightarrow$', size=18, rotation=90, ha='right', va='center')
    ax.text(0.48, -0.12, 'Best$\\longrightarrow$\nModel', size=18, rotation=90, va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.5)

    ax.set_xlabel(r'model complexity $\longrightarrow$', size=14)
    ax.set_ylabel(r'model score $\longrightarrow$', size=14)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_title("Validation Curve Schematic", size=16)

    # fig.savefig('figures/05.03-validation-curve.png')


def demo_learning_curve():
    N = np.linspace(0, 1, 1000)
    x = np.linspace(0, 1, 1000)
    y1 = 0.75 + 0.2 * np.exp(-4 * N)
    y2 = 0.7 - 0.6 * np.exp(-4 * N)

    fig, ax = plt.subplots()
    ax.plot(x, y1, lw=10, alpha=0.5, color='blue')
    ax.plot(x, y2, lw=10, alpha=0.5, color='red')

    ax.text(0.2, 0.88, "training score", rotation=-10, size=16, color='blue')
    ax.text(0.2, 0.5, "validation score", rotation=30, size=16, color='red')

    ax.text(0.98, 0.45, r'Good Fit $\longrightarrow$', size=18, rotation=90, ha='right', va='center')
    ax.text(0.02, 0.57, r'$\longleftarrow$ High Variance $\longrightarrow$', size=18, rotation=90, va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(r'training set size $\longrightarrow$', size=14)
    ax.set_ylabel(r'model score $\longrightarrow$', size=14)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_title("Learning Curve Schematic", size=16)

    # fig.savefig('figures/05.03-learning-curve.png')


def demo_naive_bayes():
    X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    ax.set_title('Naive Bayes Model', size=14)

    xlim = (-8, 8)
    ylim = (-15, 5)

    xg = np.linspace(xlim[0], xlim[1], 60)
    yg = np.linspace(ylim[0], ylim[1], 40)
    xx, yy = np.meshgrid(xg, yg)
    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

    for label, color in enumerate(['red', 'blue']):
        mask = (y == label)
        mu, std = X[mask].mean(0), X[mask].std(0)
        P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
        Pm = np.ma.masked_array(P, P < 0.03)
        ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                      cmap=color.title() + 's')
        ax.contour(xx, yy, P.reshape(xx.shape),
                   levels=[0.01, 0.1, 0.5, 0.9],
                   colors=color, alpha=0.2)

    ax.set(xlim=xlim, ylim=ylim)

    # fig.savefig('figures/05.05-gaussian-NB.png')


def demo_gaussian_basis_functions():

    class GaussianFeatures(BaseEstimator, TransformerMixin):
        """Uniformly-spaced Gaussian Features for 1D input"""

        def __init__(self, N, width_factor=2.0):
            self.N = N
            self.width_factor = width_factor

        @staticmethod
        def _gauss_basis(x, y, width, axis=None):
            arg = (x - y) / width
            return np.exp(-0.5 * np.sum(arg ** 2, axis))

        def fit(self, X, y=None):
            # create N centers spread along the data range
            self.centers_ = np.linspace(X.min(), X.max(), self.N)
            self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
            return self

        def transform(self, X):
            return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                     self.width_, axis=1)

    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = np.sin(x) + 0.1 * rng.randn(50)
    xfit = np.linspace(0, 10, 1000)

    gauss_model = make_pipeline(GaussianFeatures(10, 1.0),
                                LinearRegression())
    gauss_model.fit(x[:, np.newaxis], y)
    yfit = gauss_model.predict(xfit[:, np.newaxis])

    gf = gauss_model.named_steps['gaussianfeatures']
    lm = gauss_model.named_steps['linearregression']

    fig, ax = plt.subplots()

    for i in range(10):
        selector = np.zeros(10)
        selector[i] = 1
        Xfit = gf.transform(xfit[:, None]) * selector
        yfit = lm.predict(Xfit)
        ax.fill_between(xfit, yfit.min(), yfit, color='gray', alpha=0.2)

    ax.scatter(x, y)
    ax.plot(xfit, gauss_model.predict(xfit[:, np.newaxis]))
    ax.set_xlim(0, 10)
    ax.set_ylim(yfit.min(), 1.5)

    # fig.savefig('figures/05.06-gaussian-basis.png')


def demo_decision_tree_example():
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 0.8, 1], frameon=False, xticks=[], yticks=[])
    ax.set_title('Example Decision Tree: Animal Classification', size=24)

    def text(ax, x, y, t, size=20, **kwargs):
        ax.text(x, y, t,
                ha='center', va='center', size=size,
                bbox=dict(boxstyle='round', ec='k', fc='w'), **kwargs)

    text(ax, 0.5, 0.9, "How big is\nthe animal?", 20)
    text(ax, 0.3, 0.6, "Does the animal\nhave horns?", 18)
    text(ax, 0.7, 0.6, "Does the animal\nhave two legs?", 18)
    text(ax, 0.12, 0.3, "Are the horns\nlonger than 10cm?", 14)
    text(ax, 0.38, 0.3, "Is the animal\nwearing a collar?", 14)
    text(ax, 0.62, 0.3, "Does the animal\nhave wings?", 14)
    text(ax, 0.88, 0.3, "Does the animal\nhave a tail?", 14)

    text(ax, 0.4, 0.75, "> 1m", 12, alpha=0.4)
    text(ax, 0.6, 0.75, "< 1m", 12, alpha=0.4)

    text(ax, 0.21, 0.45, "yes", 12, alpha=0.4)
    text(ax, 0.34, 0.45, "no", 12, alpha=0.4)

    text(ax, 0.66, 0.45, "yes", 12, alpha=0.4)
    text(ax, 0.79, 0.45, "no", 12, alpha=0.4)

    ax.plot([0.3, 0.5, 0.7], [0.6, 0.9, 0.6], '-k')
    ax.plot([0.12, 0.3, 0.38], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.62, 0.7, 0.88], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.0, 0.12, 0.20], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.28, 0.38, 0.48], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.52, 0.62, 0.72], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.8, 0.88, 1.0], [0.0, 0.3, 0.0], '--k')
    ax.axis([0, 1, 0, 1])

    #fig.savefig('figures/05.08-decision-tree.png')


def demo_decision_tree_levels():
    fig, ax = plt.subplots(1, 4, figsize=(16, 3))
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1)

    X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=1.0)

    for axi, depth in zip(ax, range(1, 5)):
        model = DecisionTreeClassifier(max_depth=depth)
        mf.visualize_tree(model, X, y, ax=axi)
        axi.set_title('depth = {0}'.format(depth))

    #fig.savefig('figures/05.08-decision-tree-levels.png')


def demo_decision_tree_overfitting():
    model = DecisionTreeClassifier()

    X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=1.0)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    mf.visualize_tree(model, X[::2], y[::2], boundaries=False, ax=ax[0])
    mf.visualize_tree(model, X[1::2], y[1::2], boundaries=False, ax=ax[1])

    fig.savefig('figures/05.08-decision-tree-overfitting.png')


def demo_pca_components_rotation():
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    pca = PCA(n_components=2, whiten=True)
    pca.fit(X)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    # plot data
    ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        mf.draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
    ax[0].axis('equal');
    ax[0].set(xlabel='x', ylabel='y', title='input')

    # plot principal components
    X_pca = pca.transform(X)
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
    mf.draw_vector([0, 0], [0, 3], ax=ax[1])
    mf.draw_vector([0, 0], [3, 0], ax=ax[1])
    ax[1].axis('equal')
    ax[1].set(xlabel='component 1', ylabel='component 2',
              title='principal components',
              xlim=(-5, 5), ylim=(-3, 3.1))

    # fig.savefig('figures/05.09-PCA-rotation.png')


def demo_expectation_maximization_kmeans():

    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)

    rng = np.random.RandomState(42)
    centers = [0, 4] + rng.randn(4, 2)

    def draw_points(ax, c, factor=1):
        ax.scatter(X[:, 0], X[:, 1], c=c, cmap='viridis',
                   s=50 * factor, alpha=0.3)

    def draw_centers(ax, centers, factor=1, alpha=1.0):
        ax.scatter(centers[:, 0], centers[:, 1],
                   c=np.arange(4), cmap='viridis', s=200 * factor,
                   alpha=alpha)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='black', s=50 * factor, alpha=alpha)

    def make_ax(fig, gs):
        ax = fig.add_subplot(gs)
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        return ax

    fig = plt.figure(figsize=(15, 4))
    gs = plt.GridSpec(4, 15, left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    ax0 = make_ax(fig, gs[:4, :4])
    ax0.text(0.98, 0.98, "Random Initialization", transform=ax0.transAxes,
             ha='right', va='top', size=16)
    draw_points(ax0, 'gray', factor=2)
    draw_centers(ax0, centers, factor=2)

    for i in range(3):
        ax1 = make_ax(fig, gs[:2, 4 + 2 * i:6 + 2 * i])
        ax2 = make_ax(fig, gs[2:, 5 + 2 * i:7 + 2 * i])

        # E-step
        y_pred = pairwise_distances_argmin(X, centers)
        draw_points(ax1, y_pred)
        draw_centers(ax1, centers)

        # M-step
        new_centers = np.array([X[y_pred == i].mean(0) for i in range(4)])
        draw_points(ax2, y_pred)
        draw_centers(ax2, centers, alpha=0.3)
        draw_centers(ax2, new_centers)
        for i in range(4):
            ax2.annotate('', new_centers[i], centers[i],
                         arrowprops=dict(arrowstyle='->', linewidth=1))


        # Finish iteration
        centers = new_centers
        ax1.text(0.95, 0.95, "E-Step", transform=ax1.transAxes, ha='right', va='top', size=14)
        ax2.text(0.95, 0.95, "M-Step", transform=ax2.transAxes, ha='right', va='top', size=14)


    # Final E-step
    y_pred = pairwise_distances_argmin(X, centers)
    axf = make_ax(fig, gs[:4, -4:])
    draw_points(axf, y_pred, factor=2)
    draw_centers(axf, centers, factor=2)
    axf.text(0.98, 0.98, "Final Clustering", transform=axf.transAxes,
             ha='right', va='top', size=16)

    # fig.savefig('figures/05.11-expectation-maximization.png')


def demo_covariance_GMM_type():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05)

    rng = np.random.RandomState(5)
    X = np.dot(rng.randn(500, 2), rng.randn(2, 2))

    for i, cov_type in enumerate(['diag', 'spherical', 'full']):
        model = GMM(1, covariance_type=cov_type).fit(X)
        ax[i].axis('equal')
        ax[i].scatter(X[:, 0], X[:, 1], alpha=0.5)
        ax[i].set_xlim(-3, 3)
        ax[i].set_title('covariance_type="{0}"'.format(cov_type),
                        size=14, family='monospace')
        mf.draw_ellipse(model.means_[0], model.covars_[0], ax[i], alpha=0.2)
        ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i].yaxis.set_major_formatter(plt.NullFormatter())

    # fig.savefig('figures/05.12-covariance-type.png')