import util
import model


all_proxies = ['rhillmass', 'd5e12', 'm5e12', 'd1', 'd2', 'd5', 'd10', 's1',
               's2', 's5', 's10']
dat0 = np.load('data/darksky/galaxies_z0.npy')
dat3 = np.load('data/darksky/galaxies_z3.npy')

d0 = util.load_proxies(dat0, 'data/darksky/0/', all_proxies, all_proxies)
d3 = util.load_proxies(dat3, 'data/darksky/3/', all_proxies, all_proxies)

d0 = util.add_column(d0, 'log_c', np.log10(d0['c']))
d3 = util.add_column(d3, 'log_c', np.log10(d3['c']))

targets = ['log_c', 'spin', 'spin_bullock', 'a_half']
heatmaps = []
for dat in [dat0, dat3]:
    heatmap = []
    for target in targets:
        weights = model.getFeatureImportance(dat, all_proxies, target)
        weights /= np.mean(weights)
        heatmap.append(weights)
    heatmaps.append(heatmap)


def importance_heatmap(heatmap, proxies, targets):
    labels = [util.label_from_proxy_name(proxy) for proxy in proxies]
    plt.imshow(heatmap)
    plt.ylabel(targets)
    plt.xlabel(labels)
    plt.show()

