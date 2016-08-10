import calc as c
import util

names = ['HW', 'Becker', 'Lu', 'Henriques', 'Illustris', 'EAGLE', 'MB-II']
data_dirs = [''.join(['data/', name, '/']) for name in names]

cats = [util.get_catalog(name) for name in names]

for name, cat, ddir in zip(names, cats, data_dirs):
    df = cat['dat']
    print name
    print df.dtype.names
    proxies = ['d5e12', 'm5e12']
    proxyname = 'dm5e12'
    df = util.load_proxies(df, ddir, proxies, proxies)
    print df.dtype.names
    features = proxies + ['mstar']
    fname = 'dm5e12_all_2.dat'

    util.train_and_dump_rwp_bins(df, features, fname, proxyname, name,
                                 cat['box_size'], red_cut=cat['red_cut'],
                                 logging=True, num_splits=1)
