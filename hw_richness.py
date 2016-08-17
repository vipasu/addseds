import model, calc, util
import plotting as p
from collections import defaultdict
import numpy as np

cat = util.get_catalog('HW')
dat = cat['dat']
halo_proxies = ['rhillmass', 'd5e12', 'm5e12']
dat = util.load_proxies(dat, 'data/HW/', halo_proxies, halo_proxies)
_, rhill_df, _ = model.trainRegressor(dat, cat['box_size'],
        halo_proxies[:1])
_, dm5e12_df, _ = model.trainRegressor(dat, cat['box_size'],
        halo_proxies[1:])

mbins = [7e13, 1e14, 3e14]

def get_richness_count(df, mbins, color='ssfr', red_cut=-11):
    hosts = df[df['upid'] == -1]
    host_id_to_host_mass = {_ for _ in zip(hosts['id'], hosts['mvir'])}
    count_bins = [[],[]]
    for hid, hmass in zip(hosts['id'], hosts['mvir']):
        if hmass < mbins[0] or hmass > mbins[-1]:
            continue
        subs = df[df['upid'] == hid]
        red_subs = len(subs[color] < red_cut)
        mbin_idx = np.digitize(hmass, mbins)[0]
        count_bins.append(red_subs)
    return count_bins

hw_richness = get_richness_count(rhill_df, mbins)
rhill_richness = get_richness_count(rhill_df, mbins, 'pred')
dm5e12_richness = get_richness_count(dm5e12_df, mbins, 'pred')

def plot_richness(counts, mbins, color, label):
    masses = np.sqrt(mbins[1:] * mbins[:-1])
    means = [np.mean(bin) for bin in counts]
    scatters = [np.std(bin) for bin in counts]
    plt.errorbar(masses, means, yerr=scatters, color=color, label=label, alpha=0.7)

counts = [hw_richness, rhill_richness, dm5e12_richness]
colors = ['k', 'green', 'purple']
labels = ['HW', 'R$_{\mathrm{hill-mass}}$', '(D,M)$_{\mathrm{mass}}$']

for count, col, lab in zip(counts, colors, labels):
    plot_richness(count, mbins, col, label)
