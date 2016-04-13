import os
import cPickle as pickle
import numpy as np
import errno
import pandas as pd
from functools import reduce


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

    d_train = d.sample(frac=fraction, random_state=5432)
    d_test = d[~d.index.isin(d_train.index)]
    d_train = d_train.reset_index(drop=True)
    d_test = d_test.reset_index(drop=True)
    return d_train, d_test


def split_octant(d, box_size):
    d_octant = d[(d['x'] < box_size/2) & (d['y'] < box_size/2) & (d['z'] < box_size/2)]
    d_rest = d[~d.index.isin(d_octant.index)]
    d_octant = d_octant.reset_index(drop=True)
    d_rest = d_rest.reset_index(drop=True)
    return d_octant, d_rest


def scale_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data), scaler


def select_features(features, dataset, scaled=True):
    x_cols = [dataset[feature].values for feature in features]
    Xtot = np.column_stack(x_cols)
    y = dataset['ssfr'].values
    if scaled:
        Xtot, x_scaler = scale_data(Xtot)
        y, y_scaler = scale_data(y)
        return Xtot, y, x_scaler, y_scaler
    else:
        return Xtot, y


def pre_process(features, target, d, seed=5432):
    """
    Given features x1 and x2, and the galaxy catalog d, shuffle the data and
    then split into train and testing sets.
    """
    # machine learning bit
    N_points = len(d)
    # get the features X and the outputs y
    Xtot = np.column_stack(features)

    Xtot, x_scaler = scale_data(Xtot)
    y, y_scaler = scale_data(target)

    np.random.seed(seed)
    shuffle = np.random.permutation(N_points)
    Xtot = Xtot[shuffle]
    y = y[shuffle]

    split = int(N_points * .5)
    Xtrain, Xtest = Xtot[:split, :], Xtot[split:, :]
    ytrain, ytest = y[:split], y[split:]
    d_train, d_test = d.ix[shuffle[:split]], d.ix[shuffle[split:]]
    return Xtrain, Xtest, ytrain, ytest, d_train, d_test, x_scaler, y_scaler


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
                exclude_ids = set(exclude.id.values)
                results.append(include[include.apply(lambda x: x.upid not in exclude_ids, axis=1)])
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
    r, xra, xba, xrp, xbp, vr, vb = results
    a_red = [xra, vr]
    a_blue = [xba, vb]
    p_red = [xrp, []]
    p_blue = [xbp, []]
    return r, a_red, a_blue, p_red, p_blue


def get_wprp_bin_data(name, log_dir):
    results = load_data(name, log_dir)
    r, ssfr, pred, errs = results
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

