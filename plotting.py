import matplotlib.pyplot as plt
import seaborn as sns
import util
from matplotlib import rc
from mpl_toolkits.axes_grid1 import Grid
import numpy as np



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


def plot_rwp_split(name, log_dir, ax=None, r_scaled=True):
    ### Load the data
    #r, a_xis, a_vars, p_xis, p_vars = util.get_wprp_data(name, log_dir)
    r, a_red, a_blue, p_red, p_blue = util.get_rwp_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    ax.set_xscale('log')
    if r_scaled:
        plot_rwp(r, a_red[0], a_red[1], ax, red_col)
        plot_rwp(r, a_blue[0], a_blue[1], ax, blue_col)
        plot_rwp(r, p_red[0], p_red[1], ax, red_col)
        plot_rwp(r, p_blue[0], p_blue[1], ax, blue_col)
        ax.set_ylabel('$r$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
    else:
        plot_wprp(r, a_red[0], a_red[1], ax, red_col)
        plot_wprp(r, a_blue[0], a_blue[1], ax, blue_col)
        plot_wprp(r, p_red[0], p_red[1], ax, red_col)
        plot_wprp(r, p_blue[0], p_blue[1], ax, blue_col)
        ax.set_ylabel('$w_p(r_p)$')
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
    ax.set_xlim(9e-2, 30)
    ax.set_ylim(0, 590)
    return style_plots(ax)

def plot_rwp_bins(name, log_dir, ax=None, fill=False):
    r, actual, pred = util.get_wprp_bin_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    colors = sns.blend_palette([red_col, blue_col], len(actual))
    for col, dat in zip(colors, actual):
        xi, var = dat
        plot_rwp(r, xi, var, ax, True, '-', col, fill)
    for col, dat in zip(colors, pred):
        xi, var = dat
        plot_rwp(r, xi, var, ax, False, '--', col, fill)
    ax.set_ylabel('$r$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
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
        ax.set_yscale('log', nonposy='clip')
        ax.loglog(masses, dat[0], color=red_col, label='Actual')
        ax.loglog(masses, dat[1], color=blue_col)
        ax.fill_between(masses, dat[0]-dat[4], dat[0]+dat[4], alpha=0.5, color=red_col)
        ax.fill_between(masses, dat[1]-dat[5], dat[1]+dat[5], alpha=0.5, color=blue_col)
        ax.loglog(masses, dat[2], '--', color=red_col, label='Predicted', alpha=0.6)
        ax.loglog(masses, dat[3], '--', color=blue_col, alpha=0.6)

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


def plot_density_profile(r, m, ax):
    num_red, num_blue, num_pred_red, num_pred_blue = m
    ax.loglog(r, num_red, '--', color=red_col, lw=4, label='input')
    ax.loglog(r, num_pred_red, '--', color=red_col, label='pred', alpha=0.5)
    ax.loglog(r, num_blue, '--', color=blue_col, lw=4, label='input')
    ax.loglog(r, num_pred_blue, '--', color=blue_col, label='pred', alpha=0.5)
    ax.set_xlim(9e-2, 6)
    ax.set_ylim(5e-4, 2e1)
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
    ax.set_ylabel(r'$n(r)$')
    return style_plots(ax)


def plot_density_profile_grid(name, log_dir):
    fnames = [''.join([name, desc, '.dat']) for desc in ['_all', '_quenched', '_sf']]
    fig = plt.figure(figsize=(17,10.5))
    grid = Grid(fig, rect=111, nrows_ncols=(3,3), axes_pad=0, label_mode='L')
    for row, name in enumerate(fnames):
        r, m1, m2, m3 = util.get_density_profile_data(name, log_dir)
        for i, m in enumerate([m1, m2, m3]):
            plot_density_profile(r, m, grid[row * 3 + i])
    return grid


def annotate_density(grid, label='Text'):
    grid[8].text(1.3e-1, 5e-3, label, fontsize=40)
    ml = '\mathrm{log} M_{\mathrm{vir}}'

    # mass_labels = [''.join(['$',str(m-.25), '<', ml, '<', str(m+.25), '$']) for m in [12, 13, 14]]
    mass_labels = [''.join(['$|', ml, '-', str(m), '| < 0.25$']) for m in [12,13,14]]
    ml_pos = [(0.3, 0.2), (0.12, 1.2e-3), (0.12, 1.2e-3)]
    for i, label in enumerate(mass_labels):
        #grid[i].text(.11, 8e-4, label, fontsize=20)
        x,y = ml_pos[i]
        grid[i].text(x, y, label, fontsize=25)

    desc_labels = [''.join([name, ' Centrals']) for name in ['All', 'Quenched', 'SF']]
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
    colors = [red_col, blue_col, sns.xkcd_rgb['black']]
    labels = ['Centrals', 'Satellites', 'Combined']

    for fq, color, label in zip(dats, colors, labels):
        ax.plot(masses, fq[0], label=label, color=color)
        ax.fill_between(masses, fq[0] - fq[2], fq[0] + fq[2], color=color, alpha=0.3)
        ax.plot(masses, fq[1], '--', color=color, alpha=0.6)
    ax.legend(loc=8)

    return style_plots(ax)


def plot_quenched_fraction_vs_density(name, log_dir, ax=None):
    cutoffs, d, actual_fq, pred_fq = util.get_fq_vs_d_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
    ax.set_xlabel('$\mathrm{log}$ $\Sigma_5$')
    ax.set_ylabel('$\mathrm{Quenched}$' + ' $\mathrm{Fraction}$')
    ax.set_ylim(-0.15, 1.2)
    ax.set_xlim(-0.6, 1.3)
    colors = sns.blend_palette([blue_col, red_col], len(cutoffs) -1)
    for fq, cut, col in zip(actual_fq, cutoffs[1:], colors):
        ax.plot(d, fq, color=col, lw=3, label='$M_* < ' + str(cut) + '$')
    for fq, cut, col in zip(pred_fq, cutoffs[1:], colors):
        ax.plot(d, fq, '--', color=col, alpha=0.6)
    ax.legend(loc='best')
    return style_plots(ax)

