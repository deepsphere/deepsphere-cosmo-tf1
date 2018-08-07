"""Plotting module."""

from __future__ import division

from builtins import range
import datetime

import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils


def plot_filters_gnomonic(filters, order=10, ind=0, title='Filter {}->{}', graticule=False):
    """Plot all filters in a filterbank in Gnomonic projection."""
    nside = hp.npix2nside(filters.G.N)
    reso = hp.pixelfunc.nside2resol(nside=nside, arcmin=True) * order / 100
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
    cm = plt.cm.seismic
    cm.set_under('w')
    a = max(abs(maps.min()), maps.max())
    ymin, ymax = -a,a
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, :]
            hp.gnomview(map.flatten(), nest=True, rot=rot, reso=reso, sub=(nrows, ncols, col+row*ncols+1),
                    title=title.format(row, col), notext=True,  min=ymin, max=ymax, cbar=False, cmap=cm,
                    margins=[0.003,0.003,0.003,0.003],)
            # if row == nrows - 1:
            #     #axes[row, col].xaxis.set_ticks_position('top')
            #     #axes[row, col].invert_yaxis()
            #     axes[row, col].set_xlabel('out map {}'.format(col))
            # if col == 0:
            #     axes[row, col].set_ylabel('in map {}'.format(row))
    # fig.suptitle('Gnomoinc view of the {} filters in the filterbank'.format(filters.n_filters))#, y=0.90)
    # return fig
    if graticule:
        with utils.HiddenPrints():
            hp.graticule(verbose=False)



def plot_filters_section(filters,
                         order=10,
                         xlabel='out map {}',
                         ylabel='in map {}',
                         title='Sections of the {} filters in the filterbank',
                         figsize=None,
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

    if figsize==None:
        figsize = (12, 12/ncols*nrows)
        print(ncols, nrows)

    # Plot everything.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False, sharex='col', sharey='row')

    ymin, ymax = 1.05*maps.min(), 1.05*maps.max()
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, index_equator]
            axes[row, col].plot(angle, map, **kwargs)
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

def plot_with_std(x, y=None, color=None, alpha=0.2, ax=None, **kwargs):
    if y is None:
        y = x
        x = np.arange(y.shape[1])
    ystd = np.std(y, axis=0)
    ymean = np.mean(y, axis=0)
    if ax is None:
        ax = plt.gca()
    lines = ax.plot(x, ymean, color=color, **kwargs)
    color = lines[0].get_color()
    ax.fill_between(x, ymean - ystd, ymean + ystd, alpha=alpha, color=color)
    return ax


def zoom_mollview(sig, cmin=None, cmax=None, nest=True):
    from numpy.ma import masked_array
    from matplotlib.patches import Rectangle

    if cmin is None:
        cmin = np.min(sig)
    if cmax is None:
        cmax = np.max(sig)

    projected = hp.mollview(sig, return_projected_map=True, nest=nest)
    plt.clf()
    nmesh = 400
    loleft = -35
    loright = -30

    grid = hp.cartview(sig, latra=[-2.5,2.5], lonra=[loleft,loright], fig=1, xsize=nmesh, return_projected_map=True, nest=nest)
    plt.clf()

    nside = hp.npix2nside(len(sig))

    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))


    # Get position for the zoom window
    theta_min = 87.5/180*np.pi
    theta_max = 92.5/180*np.pi
    delta_theta = 0.55/180*np.pi
    phi_min = (180 - loleft)/180.0*np.pi
    phi_max = (180 - loright)/180.0*np.pi
    delta_phi = 0.55/180*np.pi

    angles = np.array([theta, phi]).T

    m0 = np.argmin(np.sum((angles - np.array([theta_max, phi_max]))**2, axis=1))
    m1 = np.argmin(np.sum((angles - np.array([theta_max, phi_min]))**2, axis=1))
    m2 = np.argmin(np.sum((angles - np.array([theta_min, phi_max]))**2, axis=1))
    m3 = np.argmin(np.sum((angles - np.array([theta_min, phi_min]))**2, axis=1))

    proj = hp.projector.MollweideProj(xsize=800)

    m0 = proj.xy2ij(proj.vec2xy(hp.pix2vec(ipix=m0, nside=nside)))
    m1 = proj.xy2ij(proj.vec2xy(hp.pix2vec(ipix=m1, nside=nside)))
    m2 = proj.xy2ij(proj.vec2xy(hp.pix2vec(ipix=m2, nside=nside)))
    m3 = proj.xy2ij(proj.vec2xy(hp.pix2vec(ipix=m3, nside=nside)))

    width = m0[1] - m1[1]
    height = m2[0] - m1[0]

    test_pro = np.full(shape=(400, 1400), fill_value=-np.inf)
    test_pro_1 = np.full(shape=(400, 1400), fill_value=-np.inf)
    test_pro[:,:800] = projected
    test_pro_1[:,1000:1400] = grid.data
    tt_0 = masked_array(test_pro, test_pro<-1000)
    tt_1 = masked_array(test_pro_1, test_pro_1<-1000)

    fig = plt.figure(frameon=False, figsize=(12,8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax = fig.gca()
    plt.plot(np.linspace(m1[1]+width, 1000), np.linspace(m1[0], 0), 'k-')
    plt.plot(np.linspace(m1[1]+width, 1000), np.linspace(m2[0], 400), 'k-')
    plt.vlines(x=[1000, 1399], ymin=0, ymax=400)
    plt.hlines(y=[0,399], xmin=1000, xmax=1400)

    c = Rectangle((m1[1], m1[0]), width, height, color='k', fill=False, linewidth=3, zorder=100)
    ax.add_artist(c)
    cm = plt.cm.Blues
    cm = plt.cm.RdBu_r
#     cm = plt.cm.coolwarm # Not working, I do not know why it is not working
#     cm.set_bad("white")
    im1 = ax.imshow(tt_0, cmap=cm, vmin=cmin, vmax=cmax)
    cbaxes1 = fig.add_axes([0.08,0.2,0.4,0.04])
    cbar1 = plt.colorbar(im1, orientation="horizontal", cax=cbaxes1)
    im2 = ax.imshow(tt_1, cmap=cm, vmin=cmin, vmax=cmax)
    cbaxes2 = fig.add_axes([1.02,0.285,0.025,0.43])
    cbar2 = plt.colorbar(im2, orientation="vertical", cax=cbaxes2)
    plt.xticks([])
    plt.yticks([])
    return fig


def plot_loss(loss_training, loss_validation, t_step, eval_frequency):

    x_step = np.arange(len(loss_training)) * eval_frequency
    x_time = t_step * x_step
    x_time = [datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec) for sec in x_time]

    fig, ax_step = plt.subplots()
    ax_step.semilogy(x_step, loss_training, '.-', label='training')
    ax_step.semilogy(x_step, loss_validation, '.-', label='validation')
    ax_time = ax_step.twiny()
    ax_time.semilogy(x_time, loss_training, linewidth=0)

    fmt = mpl.dates.DateFormatter('%H:%M:%S')
    ax_time.xaxis.set_major_formatter(fmt)

    ax_step.set_xlabel('Training step')
    ax_time.set_xlabel('Training time [s]')
    ax_step.set_ylabel('Loss')
    ax_step.grid(which='both')
    ax_step.legend()
