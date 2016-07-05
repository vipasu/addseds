import matplotlib.pyplot as plt
import seaborn as sns
import util
from matplotlib import rc
from mpl_toolkits.axes_grid1 import Grid
import numpy as np
import matplotlib.gridspec as gridspec



## http://python4mpia.github.io/plotting/advanced.html
## Make sure to close figures
## ax = fig.add_subplot(1, 1, 1)
## ax.set_xticks([0.1, 0.5, 0.7]) #custom placed tick marks
## ax.set_xticklabels(['a', 'b', 'c']) or ax.set_xticklabels(' ')
## ax.set_title

## major_ticks = np.arange(0, 101, 20)
## minor_ticks = np.arange(0, 101, 5)
## ax.set_xticks(major_ticks)
## ax.set_xticks(minor_ticks, minor=True)

## Removing y-axis' labels
## ax.set_yticklabels('',visible=False)

## pad the bottom
## ax.tick_params(axis='x', pad=8)

## plt.rcParams['axes.linewidth'] = 1.5
## plt.rcParams['xtick.major.size'] = 8
## plt.rcParams['xtick.minor.size'] = 4
## plt.rcParams['ytick.major.size'] = 6
## plt.rcParams['ytick.minor.size'] = 3
## plt.rcParams['ytick.minor.size'] = 3

## usage with all the params
## ax1.tick_params(axis='x',which='major',direction='out',length=4,width=4,color='b',pad=10,labelsize=20,labelcolor='g')


red_col, blue_col = sns.xkcd_rgb['reddish'], sns.xkcd_rgb['blue']

## For use with temporary contexts
def get_plotting_context():
    return sns.plotting_context('poster', font_scale=3.5,
                                rc={'xtick.labelsize': 25,
                                    'ytick.labelsize': 25,
                                    'legend.fontsize': 25})

def set_plotting_context():
    sns.set(font_scale=3.5, rc={'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25})
    sns.set_style('ticks')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)


def style_plots(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.tick_params(which='minor',axis='both',color='k',length=4,width=1, direction='in')
    # labelsize=x, pad=y
    ax.tick_params(which='major',axis='both',color='k',length=12,width=1, direction='in', pad=10)
    plt.minorticks_on()
    return ax


def plot_wprp(r, xi, var, ax, color):
    if len(var) > 0:
        ax.loglog(r, xi, '-', color=color)
        ax.fill_between(r, (xi - var), (xi + var), color=color, alpha=0.5)
    else:
        ax.loglog(r, xi, '--', color=color, alpha=0.6)
    return style_plots(ax)


def plot_rwp(r, xi, var, ax, color):
    if len(var) > 0:
        ax.plot(r, r * xi, '-', color=color)
        ax.fill_between(r, r * (xi - var), r * (xi + var), color=color, alpha=0.5)
    else:
        ax.plot(r, r * xi, '--', color=color, alpha=0.6)
    return style_plots(ax)


def plot_rwp_split_truth(name, log_dir, ax=None, r_scaled=True):
    r, a_red, a_blue, p_red, p_blue, chi2 = util.get_rwp_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    ax.set_xscale('log')
    a = 0.5
    if r_scaled:
        ax.plot(r, r *a_red[0], color='k', alpha=a)
        ax.plot(r, r *a_blue[0], color='k', alpha=a)
        ax.set_ylabel('$r_p$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
    else:
        ax.plot(r, a_red[0], color='k', alpha=a)
        ax.plot(r, a_blue[0], color='k', alpha=a)
        ax.set_ylabel('$w_p(r_p)$')
    ax.set_xlabel('$r_p$ $[Mpc$ $h^{-1}]$')
    ax.set_xlim(9e-2, 30)
    ax.set_ylim(0, 590)
    return style_plots(ax)


def plot_rwp_split(name, log_dir, ax=None, r_scaled=True):
    ### Load the data
    #r, a_red, a_blue, p_red, p_blue = util.get_rwp_data(name, log_dir)
    r, a_red, a_blue, p_red, p_blue, chi2 = util.get_rwp_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    ax.set_xscale('log')
    if r_scaled:
        plot_rwp(r, a_red[0], a_red[1], ax, red_col)
        plot_rwp(r, a_blue[0], a_blue[1], ax, blue_col)
        plot_rwp(r, p_red[0], p_red[1], ax, red_col)
        plot_rwp(r, p_blue[0], p_blue[1], ax, blue_col)
        ax.set_ylabel('$r_p$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
    else:
        plot_wprp(r, a_red[0], a_red[1], ax, red_col)
        plot_wprp(r, a_blue[0], a_blue[1], ax, blue_col)
        plot_wprp(r, p_red[0], p_red[1], ax, red_col)
        plot_wprp(r, p_blue[0], p_blue[1], ax, blue_col)
        ax.set_ylabel('$w_p(r_p)$')
    ax.set_xlabel('$r_p$ $[Mpc$ $h^{-1}]$')
    ax.set_xlim(9e-2, 30)
    ax.set_ylim(0, 590)
    print chi2
    return style_plots(ax)

def plot_rwp_bins(name, log_dir, ax=None, fill=False):
    r, actual, pred, errs = util.get_wprp_bin_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    colors = sns.blend_palette([red_col, blue_col], len(actual))
    for col, xi, var in zip(colors, actual, errs):
        plot_rwp(r, xi, var, ax, col)
    for col, xi in zip(colors, pred):
        plot_rwp(r, xi, [], ax, col)
    ax.set_ylabel('$r_p$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$', fontsize=27)
    ax.set_xlabel('$r_p$ $[Mpc$ $h^{-1}]$', fontsize=27)
    ax.set_xscale('log')
    ax.set_xlim(9e-2, 30)
    ax.set_ylim(0, 1390)
    return style_plots(ax)


def plot_HOD(name, log_dir):
    masses, f1, f2, f3, f4 = util.get_HOD_data(name, log_dir)
    fig = plt.figure(figsize=(11,11))
    grid = Grid(fig, rect=111, nrows_ncols=(2,2), axes_pad=0, label_mode='L')
    #p_scale = 8./7
    #p_c = [counts * p_scale for counts in p_c]
    #p_s = [counts * p_scale for counts in p_s]

    ax1, ax2, ax3, ax4 = grid
    labels = ['Total', 'SF/Quenched', 'Centrals', 'Satellites']
    for ax, lab in zip(grid, labels):
        ax.xaxis.labelpad = 17
        ax.set_xlabel('$M_{vir}/ M_\odot$')
        ax.set_ylabel('$N(M)$')
        ax.set_xlim(5e11, 1.7e15)
        ax.set_ylim(9e-3, 3e3)
        #ax.legend(loc=2, fontsize=20)
        ax.text(1e12, 1e2, lab, fontsize=36)
        style_plots(ax)

    ax1.loglog(masses, f1[0], color='k', label='Actual')
    ax1.loglog(masses, f1[1], '--', color='k', label='Predicted', alpha=0.6)
    ax1.fill_between(masses, f1[0] - f1[2], f1[0]+f1[2], color='k', alpha=0.5)

    for ax, dat in zip([ax2,ax3,ax4], [f2,f3,f4]):
        ax.loglog(masses, dat[0], color=red_col, label='Actual')
        ax.loglog(masses, dat[1], color=blue_col)
        ax.loglog(masses, dat[2], '--', color=red_col, label='Predicted', alpha=0.6)
        ax.loglog(masses, dat[3], '--', color=blue_col, alpha=0.6)
        ax.fill_between(masses, dat[0]-dat[4], dat[0]+dat[4], alpha=0.5, color=red_col)
        ax.fill_between(masses, dat[1]-dat[5], dat[1]+dat[5], alpha=0.5, color=blue_col)
        ax.set_yscale('log', nonposy='clip')
        ax.set_xscale('log', nonposx='clip')

    labels = ['Total', 'SF/Quenched', 'Centrals', 'Satellites']
    for ax, lab in zip(grid, labels):
        ax.xaxis.labelpad = 17
        ax.set_xlabel('$M_{vir}/ M_\odot$')
        ax.set_ylabel('$N(M)$')
        ax.set_xlim(5e11, 1.7e15)
        ax.set_ylim(9e-3, 3e3)
        #ax.legend(loc=2, fontsize=20)
        ax.text(1e12, 1e2, lab, fontsize=36)
        style_plots(ax)

    return grid

def plot_quenched_profile(r,m,ax):
    num_red, num_blue, num_pred_red, num_pred_blue, red_err, blue_err = m
    fq_actual = num_red/(num_red+num_blue)
    fq_pred = num_pred_red/(num_pred_red+num_pred_blue)
    ax.semilogx(r,fq_actual, color=red_col)
    ax.semilogx(r,fq_pred, '--', color=red_col, alpha=0.5)

    mean_err = np.sqrt((num_red * red_err)**2 + (num_blue * blue_err)**2)/(num_red + num_blue)
    ax.fill_between(r, fq_actual - mean_err, fq_actual + mean_err, color=red_col, alpha=0.3)
    ax.set_xlim(9e-2, 6)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
    ax.set_ylabel('Quenched Fraction', fontsize=32)
    ax.minorticks_on()
    return style_plots(ax)



def plot_radial_profile(r, m, ax):
    num_red, num_blue, num_pred_red, num_pred_blue, red_err, blue_err = m
    ax.semilogx(r, num_red, color=red_col, label='input')
    ax.semilogx(r, num_pred_red, '--', color=red_col, label='pred', alpha=0.5)
    ax.semilogx(r, num_blue, color=blue_col, label='input')
    ax.semilogx(r, num_pred_blue, '--', color=blue_col, label='pred', alpha=0.5)
    ax.fill_between(r, np.maximum(1e-9,num_red - red_err), num_red + red_err, color=red_col, alpha=0.3)
    ax.fill_between(r, np.maximum(1e-9,num_blue - blue_err), num_blue + blue_err, color=blue_col, alpha=0.3)
    ax.set_xlim(9e-2, 6)
    #ax.set_ylim(5e-5, 2e1)
    ax.set_ylim(0, 8.7)
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
    ax.set_ylabel(r'$N_{sat}(r)$')
    return style_plots(ax)


def plot_radial_profile_grid(name, log_dir, frac=False):
    fnames = [''.join([name, desc, '.dat']) for desc in ['_all', '_quenched', '_sf']]
    #fnames = [''.join([name, desc, '.dat']) for desc in ['_quenched', '_sf']]
    nrows = len(fnames)
    ncols = 3
    fig = plt.figure(figsize=(17,3.5 * nrows + 1.5))
    grid = Grid(fig, rect=111, nrows_ncols=(nrows,ncols), axes_pad=0, label_mode='L')
    for row, name in enumerate(fnames):
        r, m1, m2, m3 = util.get_radial_profile_data(name, log_dir)
        for i, m in enumerate([m1, m2, m3]):
            if frac:
                plot_quenched_profile(r, m, grid[row * 3 + i])
            else:
                plot_radial_profile(r, m, grid[row * 3 + i])
    return grid


def plot_rwp_bins_grid(endings, log_dir, desc='msbin_4', figsize=None):
    nrows = len(endings)
    ncols=3
    if not figsize:
        figsize = (16, 4 * nrows + 1)
    fig = plt.figure(figsize=figsize)
    grid = Grid(fig, rect=111, nrows_ncols=(nrows,ncols), axes_pad=0, label_mode='L')
    for i, dat in enumerate(endings):
        fnames =  ['_'.join(filter(lambda x: x is not '', [str(n), desc, dat])) for n in xrange(3)]
        for j, name in enumerate(fnames):
            ax = grid[i*ncols + j]
            plot_rwp_bins(name, log_dir, ax)
            ax.minorticks_on()
            ax.yaxis.label.set_size(40)
            ax.xaxis.label.set_size(40)
            if j != ncols/2:
                ax.set_xlabel('')
            if i != nrows/2:
                ax.set_ylabel('')
    return grid


def annotate_rwp_msbins(grid, labels, ncols=3, fs=40, top=1430, lheight=1050):
    c1, c2, c3 = grid[0], grid[1], grid[2]
    c1.text(1.7, top, '$9.9 < \log M_*/M_\odot < 10.1$', fontsize=25, horizontalalignment='center')
    c2.text(1.7, top, '$10.1 < \log M_*/M_\odot < 10.3$', fontsize=25, horizontalalignment='center')
    c3.text(1.7, top, '$10.5 < \log M_*/M_\odot < 10.7$', fontsize=25, horizontalalignment='center')

    for i,label in enumerate(labels):
        grid[i*ncols + ncols-1].text(20, lheight, label, fontsize=fs,
                                     horizontalalignment='right')
    return grid


def annotate_conformity(grid, label='Text'):
    grid[len(grid)-1].text(1.3e-1, 0.2, label, fontsize=40)
    ml = '\mathrm{log} M_{\mathrm{vir}}'

    # mass_labels = [''.join(['$',str(m-.25), '<', ml, '<', str(m+.25), '$']) for m in [12, 13, 14]]
    mass_labels = [ '$11.75 < \log M_{\mathrm{vir}} < 12.25$',
                    '$12.75 < \log M_{\mathrm{vir}} < 13.25$',
                    '$13.75 < \log M_{\mathrm{vir}} < 14.25$']
    for i, label in enumerate(mass_labels):
        grid[i].text(.15, 1.2, label, fontsize=25)

    desc_labels = [''.join([name, ' Centrals']) for name in ['Quenched', 'SF']]
    for i, label in enumerate(desc_labels):
        grid[3 * i].text(.109, .95, label, fontsize=30)


def annotate_radial_profile(grid, label='Text'):
    grid[len(grid)-1].text(1.3e-1, .5, label, fontsize=40)
    ml = '\mathrm{log} M_{\mathrm{vir}}'

    # mass_labels = [''.join(['$',str(m-.25), '<', ml, '<', str(m+.25), '$']) for m in [12, 13, 14]]
    mass_labels = [ '$11.75 < \log M_{\mathrm{vir}}/M_\odot < 12.25$',
                    '$12.75 < \log M_{\mathrm{vir}}/M_\odot < 13.25$',
                    '$13.75 < \log M_{\mathrm{vir}}/M_\odot < 14.25$']
    for i, label in enumerate(mass_labels):
        grid[i].text(.7, 8.9, label, fontsize=23, horizontalalignment='center')

    desc_labels = [''.join([name, ' Centrals']) for name in ['All', 'Quenched', 'Star Forming']]
    for i, label in enumerate(desc_labels):
        grid[3 * i].text(.109, 4, label, fontsize=30)

def plot_quenched_fraction(name, log_dir, ax=None):
    masses, central_fq, satellite_fq, total_fq = util.get_fq_data(name, log_dir)
    dats = [central_fq, satellite_fq, total_fq]
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlabel('$M_{\mathrm{vir}} / M_{\odot}$')
    ax.set_ylabel('$\mathrm{Quenched}$' + ' $\mathrm{Fraction}$')
    ax.set_ylim(0, 1.15)
    ax.set_xlim(5e11,1.4e15)
    colors = [sns.xkcd_rgb['green'], sns.xkcd_rgb['purple'], sns.xkcd_rgb['black']]
    labels = ['Centrals', 'Satellites', 'Combined']

    for fq, color, label in zip(dats, colors, labels):
        ax.plot(masses, fq[0], label=label, color=color)
        ax.fill_between(masses, fq[0] - fq[2], fq[0] + fq[2], color=color, alpha=0.3)
        ax.plot(masses, fq[1], '--', color=color, alpha=0.6)
    ax.legend(loc=8)

    return style_plots(ax)


def plot_quenched_fraction_vs_density(name, log_dir, ax=None):
    cutoffs, d, actual_fq, pred_fq, errs = util.get_fq_vs_d_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
    ax.set_xlabel('$\mathrm{log}$ $\Sigma_5$')
    ax.set_ylabel('$\mathrm{Quenched}$' + ' $\mathrm{Fraction}$')
    ax.set_ylim(-0.15, 1.2)
    ax.set_xlim(-1.1, 1.3)
    colors = sns.color_palette("Greens", len(errs)+1)[1:]
    for i, (fq, cut, col, err) in enumerate(zip(actual_fq, cutoffs, colors, errs)):
        ax.plot(d, fq, color=col, lw=3, label=''.join(['$',str(cut),'<\log M_*/M_\odot<',str(cutoffs[i+1]), '$']))
        ax.fill_between(d, fq - err, fq + err, color=col, alpha=0.4)
    for fq, cut, col in zip(pred_fq, cutoffs[1:], colors):
        ax.plot(d, fq, '--', color=col, alpha=0.6)
    ax.legend(loc=3, fontsize=20)
    return style_plots(ax)


def plot_twin_contour(ax, dist, ssfr, extent,log=True):
    ax2 = ax.twiny()
    if log:
        d = np.log10(dist)
    else:
        d = dist
    y_range = extent[-1:-3:-1]
    x_range = extent[0:2]
    hist_range = (y_range, x_range)#(extent[-1], extent[-2], extent[0]#((-13.5, -8.5), (-2.95,1.9))
    H, xedges, yedges = np.histogram2d(add_scatter(ssfr), d, bins=40,
                                       normed=True, range=hist_range)
    hextent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    ax2.contour(H, extent=hextent, colors='k', levels=np.linspace(.1,1,6), alpha=0.8)
    ax2.set_xticklabels([''])
    ax2.set_xticks([])


def hexbin_helper(ax, dist, ssfr, C, extent, cm,cnt, xscale='log'):
    cnt = len(dist) * .0015
    hh = ax.hexbin(dist, add_scatter(ssfr), C, xscale=xscale, mincnt=cnt,
              extent=extent, cmap=cm, alpha=0.6, gridsize=30, vmin=0, vmax=1)
    ax.minorticks_on()
    return hh


def hexbin_plots(dfs, ncols=6, cnt=20):
    nrows = len(dfs)
    labels = ['$\mathrm{HW}$', 'Becker', 'Lu', 'Henriques', 'Illustris',
              '$\mathrm{EAGLE}$', '$\mathrm{MB-II}$']
    fig = plt.figure(figsize=(17.5,3.5 * nrows + 1))
    gs = gridspec.GridSpec(nrows, ncols+1, width_ratios =[ 1] * ncols + [.15])
    gs.update(wspace=0, hspace=0)
    for i, df  in enumerate(dfs):
        print i
        extent = [-2.95, 2.3, -8.5, -13.5]
        mstar_extent = [8.7, 12.5, -8.5, -13.5]
        cm = plt.get_cmap('viridis')
        central = df.central.values

        ax1 = fig.add_subplot(gs[i,0])
        ax_list = [ax1] + [fig.add_subplot(gs[i,j]) for j in xrange(1,ncols)]
        print len(df.mstar), len(df.ssfr), len(central)

        hexbin_helper(ax_list[0], 10**df.mstar, df.ssfr, central, mstar_extent, cm, cnt)
        plot_twin_contour(ax_list[0], 10**df.mstar, df.ssfr, mstar_extent)

        hh = hexbin_helper(ax_list[1], df.rhill, df.ssfr,central, extent, cm, cnt)
        plot_twin_contour(ax_list[1], df.rhill, df.ssfr, extent)

        hexbin_helper(ax_list[2], df.rhillmass, df.ssfr,central, extent, cm, cnt)
        plot_twin_contour(ax_list[2], df.rhillmass, df.ssfr, extent)

        hexbin_helper(ax_list[3], df.d5e12, df.ssfr,central, extent, cm, cnt)
        plot_twin_contour(ax_list[3], df.d5e12, df.ssfr, extent)

        hexbin_helper(ax_list[4], 10**df.d5, df.ssfr,central, extent, cm,cnt)
        plot_twin_contour(ax_list[4], 10**df.d5, df.ssfr, extent)

        hexbin_helper(ax_list[5], 10**df.s5, df.ssfr,central, extent, cm,cnt)
        plot_twin_contour(ax_list[5], 10**df.s5, df.ssfr, extent)

        ax_list[-1].text(5e-3,-9.2,labels[i], fontsize=25)

        ax1.set_ylabel('sSFR')
        for label in ax1.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        for ax in ax_list[1:]:
            ax.set_yticklabels([])

        if i != nrows-1:
            for ax in ax_list:
                ax.set_xticklabels([])
        else:
            for ax in ax_list[1:]:
                for label in ax.xaxis.get_ticklabels()[::2]:
                    label.set_visible(False)
        for ax in ax_list:
            p.style_plots(ax)

    ds = ['$M_*/M_\odot$'] + [util.label_from_proxy_name(p) for p in ['rhill', 'rhillmass', 'd5e12', 'd5', 's5']]
    for i, ax in enumerate(ax_list):
        ax.set_xlabel(ds[i])
        ax.xaxis.labelpad = 15
    cax = fig.add_subplot(gs[:, ncols])
    plt.colorbar(hh, cax=cax, label='Central Fraction')

    return fig, gs, hh, cax
