import os
import cPickle as pickle
import calc as c
import numpy as np
import errno
import pandas as pd
from scipy.stats import rankdata
from collections import defaultdict
from functools import reduce
import numpy.lib.recfunctions


def fits_to_pandas(df):
    return pd.DataFrame.from_records(df.byteswap().newbyteorder())


def filter_nans(arr):
    return np.ma.masked_array(arr, np.isnan(arr))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def split_test_train(d, box_size, fraction=0.125):
    np.random.seed(1234)
    N = len(d)
    order = np.random.permutation(N)
    train_N = int(fraction * N)

    d_train = d[order[:train_N]]
    d_test = d[order[train_N:]]
    return d_train, d_test


def scale_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data), scaler


def select_features(features, dataset, target='ssfr', scaled=True):
    x_cols = [dataset[feature] for feature in features]
    Xtot = np.column_stack(x_cols)
    y = dataset[target]
    if scaled:
        Xtot, x_scaler = scale_data(Xtot)
        y, y_scaler = scale_data(y)
        return Xtot, y, x_scaler, y_scaler
    else:
        return Xtot, y


def jackknife_octant_samples(gals, box_size):
    half_box_size = box_size/2
    def box_split(gals, x=0, y=0, z=0):
        x_sel = np.where(gals.x > half_box_size) if x else np.where(gals.x < half_box_size)
        y_sel = np.where(gals.y > half_box_size) if y else np.where(gals.y < half_box_size)
        z_sel = np.where(gals.z > half_box_size) if z else np.where(gals.z < half_box_size)
        sel = reduce(np.intersect1d, (x_sel[0], y_sel[0], z_sel[0]))
        return gals.ix[sel], gals.drop(sel)
    results = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                include, exclude = box_split(gals, x, y, z)
                exclude_ids = set(exclude.id)
                results.append(include[np.where(map(lambda x: x.upid not in exclude_ids, include))])
    return results


def get_logging_dir(cat_name, output_dir='output/'):
    out = output_dir + cat_name + '/data/'
    mkdir_p(out)
    return out


def get_plotting_dir(cat_name, output_dir='output/'):
    out = output_dir + cat_name + '/plots/'
    mkdir_p(out)
    return out


def dump_data(data, name, log_dir):
    with open(log_dir + name, 'w') as f:
        pickle.dump(data, f)


def load_data(name, log_dir):
    with open(log_dir + name, 'r') as f:
        res = pickle.load(f)
    return res


def get_wprp_data(name, log_dir):
    results = load_data(name, log_dir)
    r, ssfr, pred = results
    a_xis, a_vars = ssfr
    p_xis, p_vars = pred
    return r, a_xis, a_vars, p_xis, p_vars


def get_rwp_data(name, log_dir):
    results = load_data(name, log_dir)
    r, actual, pred, errs, chi2 = results
    a_red = [actual[0], errs[0]]
    a_blue = [actual[1], errs[1]]
    p_red = [pred[0], []]
    p_blue = [pred[1], []]
    return r, a_red, a_blue, p_red, p_blue, chi2


def get_wprp_bin_data(name, log_dir):
    results = load_data(name, log_dir)
    r, ssfr, pred, errs, chi2 = results
    return r, ssfr, pred, errs


def get_HOD_data(name, log_dir):
    results = load_data(name, log_dir)
    masses, f1, f2, f3, f4 = results
    return masses, f1, f2, f3, f4


def get_radial_profile_data(name, log_dir):
    results = load_data(name, log_dir)
    r, m1, m2, m3 = results
    return r, m1, m2, m3


def get_fq_data(name, log_dir):
    results = load_data(name, log_dir)
    m, cents, sats, totals = results
    return m, cents, sats, totals


def get_fq_vs_d_data(name, log_dir):
    results = load_data(name, log_dir)
    cutoffs, d, actual, pred, err = results
    return cutoffs, d, actual, pred, err

def load_all_cats(prefix='./'):
    dats = defaultdict(dict)
    names = ['HW', 'Becker', 'Lu', 'Henriques', 'EAGLE', 'Illustris', 'MB-II']
    dirs = [prefix + 'data/' + name + '/' for name in names]
    box_sizes = [250.0, 250.0, 250.0, 480.0, 100.0, 75.0, 100.0]
    red_cuts = [-11.00000, -11.357438057661057, -11.031297236680984, -10.849318951368332, -10.462226897478104, -10.186549574136734, -11.175638109445572]
    for name, size, data_dir, red_cut in zip(names, box_sizes, dirs, red_cuts):
        dats[name]['box_size'] = size
        dats[name]['dir'] = data_dir
        dats[name]['red_cut'] = red_cut
    return dats

def load_dat(cats, name):
    # cats[name]['dat'] = pd.read_csv(cats[name]['dir'] + 'galaxies_cut.csv')
    cats[name]['dat'] = np.load(cats[name]['dir'] + 'halos.npy')

    return


def get_catalog(name, dir_prefix='./'):
    print "Loading cat info..."
    cats = load_all_cats(dir_prefix)
    print "loading dat..."
    assert name in cats.keys()
    load_dat(cats, name)
    cat = cats[name]
    print "Discarding extra data..."
    del cats
    return cat

def match_mstar_cut(gals, box_size, msmin, msmax):
    hwcat = get_catalog('HW')
    hwdf = hwcat['dat']

    m_fid = hwdf['mstar']
    n_fid = rankdata(-m_fid)/hwcat['box_size']**3
    mfs, nfs = zip(*sorted(zip(m_fid,n_fid)))

    msmin_idx, msmax_idx = np.digitize([msmin, msmax], mfs, right=True)
    n_min, n_max = nfs[msmin_idx], nfs[msmax_idx]

    m = gals['mstar']
    n = rankdata(-m)/box_size**3
    ms, ns = zip(*sorted(zip(m,n)))
    nmin_idx, nmax_idx = np.digitize([n_min, n_max], ns, right=True)
    return ms[nmin_idx], ms[nmax_idx]


def match_number_density(dats, nd=None, mstar=None):
    # TODO: some sort of binning to accept mstar and translate to number density

    new_dats = defaultdict(dict)
    if nd is None:
        fiducial = get_catalog('HW')['dat']
        m_f = fiducial['dat']['mstar']
        n_f= rankdata(-m_f)/fiducial['box_size']**3
        nd = max(n_f)
        print nd
    for name, cat in dats.items():
        m = cat['dat']['mstar']
        n = rankdata(-m)/cat['box_size']**3
        m_s, n_s = zip(*sorted(zip(m,n)))

        idx = np.digitize(nd, n_s, right=True)
        ms_cut = m_s[min(idx, len(m_s)-1)]
        nd_cut = n_s[min(idx, len(n_s)-1)]
        print "Cut in ", name, " at ", ms_cut, " with nd: ", nd_cut

        d = cat['dat']
        new_cat = cat.copy()
        new_cat['cut'] = ms_cut
        new_cat['dat'] = d[d['mstar'] > ms_cut]

        new_dats[name] = new_cat
    return new_dats

def train_and_dump_rwp(gals, features, name, proxy, box_name, box_size, red_cut=-11, logging=True, target='ssfr'):
    import model
    log_dir = get_logging_dir(box_name)
    d_train, d_test, regressor = model.trainRegressor(gals, box_size, features,
            target, scaled=False)
    cols = [target, 'pred']
    wprp_dat = c.wprp_split(d_test, red_cut, box_size, cols=cols)
    chi2 = wprp_dat[-1]
    if logging:
        add_statistic(box_name, 'chi2_red', proxy, chi2[0])
        add_statistic(box_name, 'chi2_blue', proxy, chi2[1])
    dump_data(wprp_dat, name, log_dir)


def train_and_dump_rwp_bins(gals, features, name, proxy, box_name, box_size, num_splits=3, red_cut=-11, logging=True, target='ssfr'):
    import model
    log_dir = get_logging_dir(box_name)
    mstar_cuts = [10.0, 10.2, 10.6]
    d_train, d_test, regressor = model.trainRegressor(gals, box_size, features,
                                                      target=target, model=model.DecisionTreeRegressor, scaled=False)
    for i,cut in enumerate(mstar_cuts):
        msmin, msmax = match_mstar_cut(gals, box_size, cut-.1, cut+.1)
        print "Matching cut for ", cut-.1, cut+.1
        print "\t ", msmin, msmax

        mstar_sel = np.where((d_test['mstar'] < msmax) & (d_test['mstar'] > msmin))[0]

        if num_splits is 3:
            res = c.wprp_bins(d_test[mstar_sel],
                            num_splits, box_size)
        elif num_splits is 1:
            print "Calling wprp split with red cut of:", red_cut
            res = c.wprp_split(d_test[mstar_sel],
                            red_cut, box_size)
        chi2 = res[-1]
        if num_splits is 3:
            stat_names = ['chi2_red_' + x for x in ['75', '50', '25', '00']]
        elif num_splits is 1:
            stat_names = ['chi2_red', 'chi2_blue']
        stat_names = [sname + '_' + str(i) for sname in stat_names]
        if logging:
            for j, sname in enumerate(stat_names):
                add_statistic(box_name, sname, proxy, chi2[j])
        dump_data(res, str(i) + '_msbin_' + str(num_splits+1) + '_' + name, log_dir)

def load_feature_list(proxy, dat, cat):
    proxy_cache = {
        'dm5e12': ['d5e12', 'm5e12'],
        'dmc5e12': ['d5e12', 'm5e12', 'c5e12'],
        'sn5e12': ['s5e12', 'ns5e12'],
        }

    if proxy in proxy_cache.keys():
        features = proxy_cache[proxy]
    else:
        features = [proxy]
    return features


def load_proxies(gals, data_dir, proxy_names, dat_names):
    for proxy, name in zip(proxy_names, dat_names):
        if proxy not in gals.dtype.names:
            vals = np.load(data_dir + name + '.npy')
            nan_sel = np.where(np.isnan(vals))[0]
            if len(nan_sel) > 0:
                print "Replacing %d NaN values for %s" % (len(nan_sel), proxy)
                vals[nan_sel] = 100
            gals = add_column(gals, proxy, vals)
    return gals


def label_from_proxy_name(name):
    name_label_dict = {
        'rhill':'R$_{\mathrm{hill}}$',
        'rhillmass':'R$_{\mathrm{hill_{\mathrm{mass}}}}$',
        'dm5e12':'(D,M)$_{\mathrm{mass}}$',
        'dmc5e12':'(D,M,c)$_{\mathrm{mass}}$',
        'sn5e12':'$(\Sigma,N_{\mathrm{gal}})_{\mathrm{mass}}$',
        'd5e12':'D$_{\mathrm{mass}}$',
        'd1':'D$_{1}$',
        'd2':'D$_{2}$',
        'd5':'D$_{5}$',
        'd10':'D$_{10}$',
        's1':'$\Sigma_{1}$',
        's2':'$\Sigma_{2}$',
        's5':'$\Sigma_{5}$',
        's10':'$\Sigma_{10}$',
    }
    if name in name_label_dict.keys():
        return name_label_dict[name]
    elif name[:3] == 'rhm':
        return r'$M/M_\odot > %s \times 10^{%s}$' %(name[3], name[5:7])
    print "failed to write name"
    return None



def match_quenched_fraction(dat, f_q=0.477807721657):
    hwdat = get_catalog('HW')['dat']
    n = len(dat)
    left, right = -13.0, -9.0
    red_cut = (left + right)/2
    quenched_fraction = 1.0 * len(dat[dat.ssfr < red_cut])/n
    tol = 1e-7
    while (right-left) > tol:
        #print quenched_fraction
        if quenched_fraction < f_q:
            left = red_cut
        else:
            right = red_cut
        red_cut = (left + right)/2
        quenched_fraction = 1.0 * len(dat[dat.ssfr < red_cut])/n
        #print left, right
    return red_cut


def add_statistic(cat_name, stat_name, proxy_name, value):
    data_dir = 'data/' + cat_name + '/'
    try:
        stat_dict = load_data('statistics.pckl', data_dir)
    except:
        stat_dict = defaultdict(dict)
    if not stat_dict[stat_name].has_key(proxy_name):
        stat_dict[stat_name][proxy_name] = []
    stat_dict[stat_name][proxy_name].append(value)
    dump_data(stat_dict, 'statistics.pckl', data_dir)


def recarray_from_npz(fname):
    dat = np.load(fname)
    return np.recarray({x : dat[x] for x in dat.files})


def add_column(arr, name, vals):
    return numpy.lib.recfunctions.append_fields(arr, name, vals, usemask=False,
            asrecarray=True)


