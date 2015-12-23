import matplotlib.pyplot as plt
import seaborn as sns
import util

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
    ax.tick_params(which='major',axis='both',color='k',length=8,width=1, direction='in', pad=10)
    plt.minorticks_on()
    return ax




## TODO: Come up with data format for HOD
## TODO: Come up with data format for number density
## TODO: Come up with data format for wprp - done

def plot_rwp(name, log_dir):
    ### Load the data
    r, a_xis, a_vars, p_xis, p_vars = util.get_wprp_data(name, log_dir)
    fig = plt.figure(figsize=(12, 12))
    plt.xscale('log')
    plt.errorbar(r, r * a_xis[0], r*a_vars[0], fmt='-o', color=red_col)
    plt.errorbar(r, r * a_xis[1], r*a_vars[1], fmt='-o', color=blue_col)
    plt.errorbar(r, r * p_xis[0], r*p_vars[0], fmt='--o', color=red_col, alpha=0.6)
    plt.errorbar(r, r * p_xis[1], r*p_vars[1], fmt='--o', color=blue_col, alpha=0.6)

    ### Formatting stuff
    plt.ylabel('$r$ $w_p(r_p)$')
    plt.xlabel('$r$ $[Mpc$ $h^{-1}]$')
    plt.xlim(1e-1, 30)
    return style_plots()

#def plot_rwp_bins(fname):
