import calc as c
import util

cat_name = 'HW'
cat = util.get_catalog('HW')
dat = cat['dat']
proxy_names = [
 'rhm3e12', 'rhm4e12', 'rhm5e12', 'rhm6e12', 'rhm7e12', 'rhm8e12', 'rhm9e12',
'rhm1e13', 'rhm2e13', 'rhm3e13', 'rhm4e13', 'rhm5e13'
]
names = [proxy + '.dat' for proxy in proxy_names]
all_features = [util.load_feature_list(proxy, dat, cat) for proxy in proxy_names]
for features, name, proxy in zip(all_features, names, proxy_names):
    print "training on: ", features + ['mstar']

    util.train_and_dump_rwp_bins(dat, features + ['mstar'], name, proxy, cat_name, cat['box_size'], num_splits=3, logging=True)
