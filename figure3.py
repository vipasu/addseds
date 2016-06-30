import calc as c
import util

cat_name = 'HW'
cat = util.get_catalog('HW')
dat = cat['dat']
proxy_names = ['sn5e12', 's5']
#[ 'd5', 's5', 'rhill', 'rhillmass', 'dm5e12', 'd5e12', 'sn5e12', 'd1', 'd2', 'd10', 's1','s2','s10']
names = [proxy + '.dat' for proxy in proxy_names]
all_features = [util.load_feature_list(proxy, dat, cat) for proxy in proxy_names]
for features, name, proxy in zip(all_features, names, proxy_names):
    print "training on: ", features + ['mstar']

    try:
        util.train_and_dump_rwp_bins(dat, features + ['mstar'], name, proxy, cat_name, cat['box_size'], num_splits=3, logging=True)
    except:
        pass
