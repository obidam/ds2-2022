#!/bin/env python
# -*coding: UTF-8 -*-
#
# Useful functions for practice notebooks
#
# Created by gmaze on 04/10/2018
__author__ = 'gmaze@ifremer.fr'

import numpy as np
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature

import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="deep", color_codes=True)

# Trivial functions to create and manipulate 2D matrices [n_samples,n_features=2]

def new_sample(m,s,n=100,r=170):
    """Return a 2D normal distribution centered on m with std s"""
    X, y = make_blobs(n_samples=n, random_state=r, centers=[m], cluster_std=s)
    return {'data':X,'labels':y}

def stretch(X,m=[1,1]):
    """Stretch a 2d matrix X by factors m[0] and m[1] along dimensions 0 and 1"""
    return np.dot(X, [[m[0], 0], [0, m[1]]])

def rotate(X,a=0):
    """Rotate around the origin a 2d matrix X by angle a in degrees"""
    a = np.deg2rad(a) # Convert degrees to radians
    return np.dot(X, [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

def translate(X,v=[0,0]):
    """Translate 2d matrix X by v[0] and v[1] along dimensions 0 and 1"""
    n = X.shape[0]
    X[:,0] += np.full((n),v[0])
    X[:,1] += np.full((n),v[1])
    return X


# Misc functions

def gaussian(x, mu, sigsq):
    """Return a 1D Normal Density Function along axis x"""
    scale = 1/np.sqrt(2*np.pi*sigsq)
    return scale*np.exp(-np.power(x - mu, 2.) / (2 * sigsq))

def vrange(V):
    """Return an array value range"""
    return [np.min(V),np.max(V)]

def vrangec(V):
    """"Return an array value centered range"""
    xl = np.max(np.abs(vrange(V)))
    return np.array([-xl,xl])


# Plotting functions

def create_map(extent=[-180, 180, -70, 70], dpi=200, figsize=(12,4)):
    """Create a figure with a map
        Return fig, proj, ax
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0,0,1,1],projection=proj)
    ax.set_extent(extent, crs=proj)
    gl=ax.gridlines(crs=proj, draw_labels=True,
        linewidth=0.5, color=[0.6]*3, alpha=0.5, linestyle='--')
    # gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,360/10+1))
    # gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,180/5+1))
    gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,int(360/30)+1))
    gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,int(180/20)+1))
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cfeature.LAND, facecolor=[0.7]*3)
    ax.add_feature(cfeature.COASTLINE)
    return fig, proj, ax

def plot2d_labels(X, labels, cluster_centers=np.empty(()), dpi=80, kmarkersize=10):
    """Create figure with a scatter plot of X colored by 'labels'"""
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(0.6), dpi=90, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=dpi, facecolor='w', edgecolor='k')

    unique_labels = np.unique(labels)
    n_clusters_ = unique_labels.shape[0]
    colors = sns.husl_palette(n_clusters_)

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k

        plt.plot(X[class_members,0], X[class_members,1], '.',
                 markerfacecolor=tuple(col), markeredgecolor='none', label='Class %i'%(k))

        if len(cluster_centers.shape)>0:
            plt.plot(cluster_centers[k,0], cluster_centers[k,1], 'X', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=kmarkersize, label='Class center #%i'%(k))

    plt.axis('equal')
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.title('Number of clusters: %i' % n_clusters_)
    plt.legend()
    return fig, ax, colors

def plot_GMMellipse(gmm,ik,col,ax,label="",std=[1],main_axes=True,**kwargs):
    """
        Plot an 1-STD ellipse for a given component (ik) and 2 given dimensions (id)
        of the GMM model gmm
        This is my routine, simply working with a matplotlib plot method
        I also added the possibility to plot the main axes of the ellipse
    """
    id = [0, 1] # Dimensions id
    covariances = gmm.covariances_[ik][(id[0],id[0],id[1],id[1]),(id[0],id[1],id[0],id[1])].reshape(2,2)
    d, v = np.linalg.eigh(covariances) # eigenvectors have unit length
    d = np.diag(d)
    theta = np.arange(0,2*np.pi,0.02)
    x = np.sqrt(d[0,0])*np.cos(theta)
    y = np.sqrt(d[1,1])*np.sin(theta)
    xy = np.array((x,y)).T
    ii = 0
    for nstd in np.array(std):
        ii+=1
        ellipse = np.inner(v,xy).T
        ellipse = nstd*ellipse + np.ones((theta.shape[0], 1))*gmm.means_[ik,(id[0],id[1])]
        if ii == 1:
    #            p = ax.plot(ellipse[:,0], ellipse[:,1], color=col, axes=ax, label=("%s (%i-std)")%(label,nstd),**kwargs)
            p = ax.plot(ellipse[:,0], ellipse[:,1], color=col, axes=ax, label=("%s")%(label),**kwargs)
        else:
            p = ax.plot(ellipse[:,0], ellipse[:,1], color=col, axes=ax,**kwargs)
    if main_axes: # Add Main axes:
        for idir in range(2):
            l = np.sqrt(d[idir,idir])*v[:,idir].T
            start = gmm.means_[ik,(id[0],id[1])]-l
            endpt = gmm.means_[ik,(id[0],id[1])]+l
            linex = [start[0], endpt[0]]
            liney = [start[1], endpt[1]]
            plt.plot(linex,liney,color=col,axes=ax,**kwargs)
    return p, ax

def sns_GMMellipse(x,y,gmm=[],std=[1],main_axes=True,label="?",colors=None,**kwargs):
    """
        Same as plot_GMMellipse but works in Seaborn join plots
    """
    id = [0,1] # Dimensions
    K = gmm.n_components
#    colors = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
#     colors = sns.color_palette("Paired", K)
    for ik in np.arange(K):
#        col = next(colors)
        col = colors[ik]
        covariances = gmm.covariances_[ik][(id[0],id[0],id[1],id[1]),(id[0],id[1],id[0],id[1])].reshape(2,2)
        d, v = np.linalg.eigh(covariances) #  eigenvectors have unit length
        d = np.diag(d)
        theta = np.arange(0,2*np.pi,0.02)
        x = np.sqrt(d[0,0])*np.cos(theta)
        y = np.sqrt(d[1,1])*np.sin(theta)
        xy = np.array((x,y)).T
        ii = 0
        for nstd in np.array(std):
            ii+=1
            ellipse = np.inner(v,xy).T
            ellipse = nstd*ellipse + np.ones((theta.shape[0], 1))*gmm.means_[ik,(id[0],id[1])]
            if ii == 1:
                plt.plot(ellipse[:,0], ellipse[:,1], color=col, label=("%s")%(label),**kwargs)
            else:
                plt.plot(ellipse[:,0], ellipse[:,1], color=col, **kwargs)
        if main_axes: # Add Main axes:
            for idir in range(2):
                l = np.sqrt(d[idir,idir])*v[:,idir].T
                start = gmm.means_[ik,(id[0],id[1])]-l
                endpt = gmm.means_[ik,(id[0],id[1])]+l
                linex = [start[0], endpt[0]]
                liney = [start[1], endpt[1]]
                plt.plot(linex,liney,color=col)

def sns_plot2d_GMM_marginals(df, gmm):
    """Plot a 2D dataset pdf with marginal pdfs and GMM modes

        Return the seaborn JointGrid object

        If X is a 2d array with data and labels the 1d labels:
            df = pd.DataFrame(np.concatenate((X,labels[np.newaxis].T),axis=1), columns=["x", "y", "labels"])
            g = sns_plot2d_GMM_marginal(df, gmm)
    """

    unique_labels = np.unique(df['labels'])
    n_clusters_ = unique_labels.shape[0]
    colors = sns.husl_palette(n_clusters_)
#     xlim = vrangec(df['x'])
#     ylim = vrangec(df['y'])
    alim = np.max(np.abs(df[["x", "y"]]).max(axis=1))
    xlim = np.array([-alim,alim])
    ylim = np.array([-alim,alim])

    # Figure
    g = sns.JointGrid(x="x", y="y", data=df, size=10, ratio=4, space=0.1,
                      xlim=xlim, ylim=ylim)
    g.fig.suptitle("GMM modes (colors) and Marginal PDFs (black)", fontsize=16)

    # cmap = sns.light_palette("gray",reverse=False,as_cmap=True)
    cmap = sns.light_palette("gray", as_cmap=True)
    g.plot_joint(sns.kdeplot, kernel='gau', shade=True,
                 shade_lowest=False,
                 cmap=cmap, kind='hex', n_levels=30)
    g.plot_joint(sns.kdeplot, kernel='gau', shade=False, n_levels=10, linewidth=0.5)

    g.plot_marginals(sns.distplot, norm_hist=True, kde=False, color=".5")
    # g.plot_marginals(sns.distplot, hist=False, kde=True, kde_kws={'kernel': 'gau', 'color': 'k', 'linestyle': '--'})

    y = np.linspace(vrangec(df['y'])[0], vrangec(df['y'])[1], 200)
    gmm_pdf = np.zeros(y.shape)
    for k, col in zip(range(n_clusters_), colors):
        _ = g.ax_marg_y.plot(gmm.weights_[k] * gaussian(y, gmm.means_[k, 1], gmm.covariances_[k][1, 1]), y,
                             color=col, label='Class %i'%k)
        gmm_pdf += gmm.weights_[k] * gaussian(y, gmm.means_[k, 1], gmm.covariances_[k][1, 1])
    _ = g.ax_marg_y.plot(gmm_pdf, y, color='k')

    x = np.linspace(vrangec(df['x'])[0], vrangec(df['x'])[1], 200)
    gmm_pdf = np.zeros(x.shape)
    for k, col in zip(range(n_clusters_), colors):
        _ = g.ax_marg_x.plot(x, gmm.weights_[k] * gaussian(x, gmm.means_[k, 0], gmm.covariances_[k][0, 0]),
                             color=col, label='Class %i'%k)
        gmm_pdf += gmm.weights_[k] * gaussian(x, gmm.means_[k, 0], gmm.covariances_[k][0, 0])
    _ = g.ax_marg_x.plot(x, gmm_pdf, color='k')

    g.plot_joint(plt.scatter, c=".5", s=1, linewidth=1, marker=".")

    g.plot_joint(sns_GMMellipse, gmm=gmm, main_axes=True, linewidth=3, colors=colors)

    # g.ax_joint.legend(prop={'weight': 'bold', 'size': 12}, loc='upper left')
    g.ax_marg_x.legend(prop={'weight': 'bold', 'size': 12}, loc='upper left')

    return g

