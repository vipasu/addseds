import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotting as plots
import sys
from matplotlib.colors import LogNorm

plots.set_plotting_context()

def plot_heatmap(dat, xlabels, ylabels, name):
    num_x = len(xlabels)
    num_y = len(ylabels)
    plt.figure(figsize=(10, 10))
    plt.pcolor(dat, cmap='RdYlGn_r', norm=LogNorm())
    #plt.pcolor(dat, cmap='coolwarm')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.xticks(np.arange(num_x)+.5, xlabels, rotation='vertical', fontsize=25)
    plt.yticks(np.arange(num_y)+.5, ylabels, rotation='horizontal', fontsize=25)
    plt.subplots_adjust(bottom=0.3, left=0.3)
    plt.title(name)
    plots.style_plots()
    plt.show()

def main(stat, stat_name):
    cats = util.load_all_cats()
    all_r_values = []
    names = cats.keys()
    names = ['HW', 'Becker', 'Lu', 'Henriques', 'Illustris', 'EAGLE', 'MB-II'][::-1]
    proxies = ['s1','s2','s5','s10','d1','d2','d5','d10', 'rhill', 'rhillmass']
    proxies_formatted = [ '$\Sigma_1$', '$\Sigma_2$', '$\Sigma_5$', '$\Sigma_{10}$', '$D_1$', '$D_2$', '$D_5$', '$D_{10}$', 'R$_\mathrm{hill}$', 'R$_\mathrm{hill-mass}$' ]
    for name in names:
        cat = cats[name]
        stat_dict = util.load_data('statistics.pckl', cat['dir'])
        r_values = []
        for p in proxies:
            try:
                print 'std of ', stat,' for ', p, '=', np.std(stat_dict[stat][p])
                r_values.append(np.mean(stat_dict[stat][p]))
            except:
                print 'no statistics found for', p
                r_values.append(0)
        all_r_values.append(r_values)
    df = pd.DataFrame(columns=proxies_formatted, index=names)
    for name, r_values in zip(names, all_r_values):
        df.loc[name] = pd.Series({p: v for p,v in zip(proxies_formatted, r_values)})
    #plt.imshow(all_r_values)
    #plt.show()
    df = df[df.columns].astype(float)
    #sns.heatmap(df, vmin=0,vmax=0.71, cmap='Blues', annot=True, fmt='.2f')
    #plots.style_plots()
    #plt.show()
    print df.values
    plot_heatmap(df, proxies_formatted, names, stat_name)

if __name__ == '__main__':
    stat = sys.argv[1]
    formatted_name = sys.argv[2]
    main(stat, formatted_name)

