import matplotlib.pyplot as plt
import seaborn as sns
import util
from matplotlib import rc
from mpl_toolkits.axes_grid1 import Grid


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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


def style_plots(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.tick_params(which='minor',axis='both',color='k',length=4,width=1, direction='in')
    # labelsize=x, pad=y
    ax.tick_params(which='major',axis='both',color='k',length=12,width=1, direction='in', pad=10)
    plt.minorticks_on()
    return ax




## TODO: Come up with data format for HOD
## TODO: Come up with data format for number density
## TODO: Come up with data format for wprp - done


def plot_rwp(r, xi, var, ax, actual, fmt, color, fill):
    if fill:
        alphas = [0.5, 1] if actual else [0.2, 0.6]
        ax.fill_between(r, r * (xi - var), r * (xi + var), color=color, alpha=alphas[0])
        ax.plot(r, r * xi, fmt, color=color, alpha=alphas[1])
    else:
        alpha = 1 if actual else 0.6
        ax.errorbar(r, r * xi, r * var, fmt=fmt, color=color, alpha=alpha)
    return style_plots(ax)


def plot_rwp_split(name, log_dir, ax=None, fill=False):
    ### Load the data
    r, a_xis, a_vars, p_xis, p_vars = util.get_wprp_data(name, log_dir)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
    ax.set_xscale('log')
    plot_rwp(r, a_xis[0], a_vars[0], ax, True, '-o', red_col, fill)
    plot_rwp(r, a_xis[1], a_vars[1], ax, True, '-o', blue_col, fill)
    plot_rwp(r, p_xis[0], p_vars[0], ax, False, '--o', red_col, fill)
    plot_rwp(r, p_xis[1], p_vars[1], ax, False, '--o', blue_col, fill)
    ax.set_ylabel('$r$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
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
        plot_rwp(r, xi, var, ax, True, '-o', col, fill)
    for col, dat in zip(colors, pred):
        xi, var = dat
        plot_rwp(r, xi, var, ax, False, '--o', col, fill)
    ax.set_ylabel('$r$ $w_p(r_p)$ $[Mpc$ $h^{-1}]$')
    ax.set_xlabel('$r$ $[Mpc$ $h^{-1}]$')
    ax.set_xscale('log')
    ax.set_xlim(9e-2, 30)
    ax.set_ylim(0, 1390)
    return style_plots(ax)


def plot_HOD(name, log_dir):
    masses, num_halos, a_c, a_s, p_c, p_s = util.get_HOD_data(name, log_dir)
    fig = plt.figure(figsize=(11,11))
    grid = Grid(fig, rect=111, nrows_ncols=(2,2), axes_pad=0, label_mode='L')
    p_scale = 8./7
    p_c = [counts * p_scale for counts in p_c]
    p_s = [counts * p_scale for counts in p_s]

    ax1, ax2, ax3, ax4 = grid
    # TODO: Error bars
    total_actual = a_c[0] + a_c[1] + a_s[0] + a_s[1]
    total_pred = p_c[0] + p_c[1] + p_s[0] + p_s[1]
    ax1.loglog(masses, total_actual/num_halos, color='k', lw=4, label='Actual')
    ax1.loglog(masses, total_pred/num_halos, '--', color='k', label='Predicted', alpha=0.6)

    total_red = a_c[0] + a_s[0]
    total_blue = a_c[1] + a_s[1]
    pred_red = p_c[0] + p_s[0]
    pred_blue = p_c[1] + p_s[1]
    ax2.loglog(masses, total_red/num_halos, color=red_col, lw=4, label='Actual')
    ax2.loglog(masses, total_blue/num_halos, color=blue_col, lw=4)
    ax2.loglog(masses, pred_red/num_halos, '--', color=red_col, label='Predicted', alpha=0.6)
    ax2.loglog(masses, pred_blue/num_halos, '--', color=blue_col, alpha=0.6)

    ax3.loglog(masses, a_c[0]/num_halos, color=red_col, lw=4, label='Actual')
    ax3.loglog(masses, a_c[1]/num_halos, color=blue_col, lw=4)
    ax3.loglog(masses, p_c[0]/num_halos, '--', color=red_col, label='Predicted', alpha=0.6)
    ax3.loglog(masses, p_c[1]/num_halos, '--', color=blue_col, alpha=0.6)

    ax4.loglog(masses, a_s[0]/num_halos, color=red_col, lw=4, label='Actual')
    ax4.loglog(masses, a_s[1]/num_halos, color=blue_col, lw=4)
    ax4.loglog(masses, p_s[0]/num_halos, '--', color=red_col, label='Predicted', alpha=0.6)
    ax4.loglog(masses, p_s[1]/num_halos, '--', color=blue_col, alpha=0.6)

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

