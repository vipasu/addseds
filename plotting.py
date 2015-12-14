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
    ax.tick_params(which='minor',axis='both',color='k',length=10,width=2, direction='in')
    # labelsize=x, pad=y
    ax.tick_params(which='major',axis='both',color='k',length=20,width=2, direction='in', pad=50)
    plt.minorticks_on()
    return ax


def get_logging_dir(cat_name, output_dir='output/'):
    out = output_dir + cat_name + '/data/'
    util.mkdir_p(out)
    return out


def get_plotting_dir(cat_name, output_dir='output/'):
    out = output_dir + cat_name + '/plots/'
    util.mkdir_p(out)
    return out


## TODO: Come up with data format for HOD
## TODO: Come up with data format for number density
## TODO: Come up with data format for wprp
