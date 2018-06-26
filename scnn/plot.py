"""Plotting module."""

from __future__ import division
import numpy as np
import healpy as hp
from builtins import range
import matplotlib.pyplot as plt


def plot_filters_gnomonic(filters, order=10, ind=0, title='Filter {}->{}'):
    """Plot all filters in a filterbank in Gnomonic projection."""
    nside = hp.npix2nside(filters.G.N)
    reso = hp.pixelfunc.nside2resol(nside=nside, arcmin=True) * order / 70
    rot = hp.pix2ang(nside=nside, ipix=ind, nest=True, lonlat=True)

    maps = filters.localize(ind, order=order)

    nrows, ncols = filters.n_features_in, filters.n_features_out

    if maps.shape[0] == filters.G.N:
        # FIXME: old signal shape when not using Chebyshev filters.
        shape = (nrows, ncols, filters.G.N)
        maps = maps.T.reshape(shape)
    else:
        if nrows == 1:
            maps = np.expand_dims(maps, 0)
        if ncols == 1:
            maps = np.expand_dims(maps, 1)

    # Plot everything.
    # fig, axes = plt.subplots(nrows, ncols, figsize=(17, 17/ncols*nrows),
    #                          squeeze=False, sharex='col', sharey='row')

    # ymin, ymax = 1.05*maps.min(), 1.05*maps.max()
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, :]
            hp.gnomview(map.flatten(), nest=True, rot=rot, reso=reso, sub=(nrows, ncols, col+row*ncols+1),
                    title=title.format(row, col), notext=True)
            # if row == nrows - 1:
            #     #axes[row, col].xaxis.set_ticks_position('top')
            #     #axes[row, col].invert_yaxis()
            #     axes[row, col].set_xlabel('out map {}'.format(col))
            # if col == 0:
            #     axes[row, col].set_ylabel('in map {}'.format(row))
    # fig.suptitle('Gnomoinc view of the {} filters in the filterbank'.format(filters.n_filters))#, y=0.90)
    # return fig


def plot_filters_section(filters,
                         order=10,
                         xlabel='out map {}',
                         ylabel='in map {}',
                         title='Sections of the {} filters in the filterbank',
                         **kwargs):
    """Plot the sections of all filters in a filterbank."""

    nside = hp.npix2nside(filters.G.N)
    npix = hp.nside2npix(nside)

    # Create an inverse mapping from nest to ring.
    index = hp.reorder(range(npix), n2r=True)

    # Get the index of the equator.
    index_equator, ind = get_index_equator(nside, order)
    nrows, ncols = filters.n_features_in, filters.n_features_out

    maps = filters.localize(ind, order=order)
    if maps.shape[0] == filters.G.N:
        # FIXME: old signal shape when not using Chebyshev filters.
        shape = (nrows, ncols, filters.G.N)
        maps = maps.T.reshape(shape)
    else:
        if nrows == 1:
            maps = np.expand_dims(maps, 0)
        if ncols == 1:
            maps = np.expand_dims(maps, 1)

    # Make the x axis: angular position of the nodes in degree.
    angle = hp.pix2ang(nside, index_equator, nest=True)[1]
    angle -= abs(angle[-1] + angle[0]) / 2
    angle = angle / (2 * np.pi) * 360

    # Plot everything.
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 12/ncols*nrows),
                             squeeze=False, sharex='col', sharey='row')

    ymin, ymax = 1.05*maps.min(), 1.05*maps.max()
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, index_equator]
            axes[row, col].plot(angle, map)
            axes[row, col].set_ylim(ymin, ymax)
            if row == nrows - 1:
                #axes[row, col].xaxis.set_ticks_position('top')
                #axes[row, col].invert_yaxis()
                axes[row, col].set_xlabel(xlabel.format(col))
            if col == 0:
                axes[row, col].set_ylabel(ylabel.format(row))
    fig.suptitle(title.format(filters.n_filters))#, y=0.90)
    return fig


def plot_index_filters_section(filters, order=10, rot=(180,0,180)):
    """Plot the indexes used for the function `plot_filters_section`"""
    nside = hp.npix2nside(filters.G.N)
    npix = hp.nside2npix(nside)

    index_equator, center = get_index_equator(nside, order)

    sig = np.zeros([npix])
    sig[index_equator] = 1
    sig[center] = 2
    hp.mollview(sig, nest=True, title='', cbar=False, rot=rot)


def get_index_equator(nside, radius):
    """Return some indexes on the equator and the center of the index."""
    npix = hp.nside2npix(nside)

    # Create an inverse mapping from nest to ring.
    index = hp.reorder(range(npix), n2r=True)

    # Center index
    center = index[npix // 2]

    # Get the value on the equator back.
    equator_part = range(npix//2-radius, npix//2+radius+1)
    index_equator = index[equator_part]

    return index_equator, center

def plot_with_std(x, y=None, color='b', ax=None, **kwargs):
    if y is None:
        y = x
        x = np.arange(y.shape[1])
    ystd = np.std(y,axis=0)
    ymean = np.mean(y,axis=0)
    transparency = 0.2
    if ax is None:
        ax = plt.gca()
    ax.plot(x, ymean, color=color, **kwargs)
    ax.fill_between(x, ymean - ystd, ymean + ystd, alpha=transparency, color=color)
    return ax
