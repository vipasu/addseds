from CorrelationFunction import projected_correlation
import os
import errno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fast3tree import fast3tree
import fitsio
import util
import plotting as p

### TODO: define functionality
# API Calls
# c.wprp
# c.wprp_split
# c.HOD
# c.density_profile

# desired checks
# make sure everyone has a position
# has mstar
# has ssfr
# has id

h = 0.7
L = 250.0/h
zmax = 40.0

rpmin = 0.1
rpmax = 20.0
Nrp = 25
rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
r = np.sqrt(rbins[1:]*rbins[:-1])

## TODO: define a catalog class


def make_pos(gals, pos_tags=['x', 'y', 'z']):
    pos = np.zeros((len(gals), 3))
    for i, tag in enumerate(pos_tags):
        pos[:, i] = gals[tag][:]
    return pos


def make_r_scale(rmin, rmax, Nrp):
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), Nrp+1)
    r = np.sqrt(rbins[1:]*rbins[:-1])
    return r, rbins


def catalog_selection(d0, m, msmin, msmax=None):
    """
    Create host set with mvir > m
    and galaxy set with mstar in the range (msmin, msmax)
    """
    # get parent halo set
    hosts = d0[(d0.upid == -1) & (d0.mvir >= m)]
    hosts = hosts.reset_index(drop=True)

    # make stellar mass bin
    df = d0[d0.mstar >= msmin]
    if msmax is not None:
        df = df[df.mstar < msmax]
    df = df.reset_index(drop=True)
    return df, hosts


def get_projected_dist_and_attrs(hosts, gals, nn, attrs, box_size=250.0):
    width_by_2 = 0.005
    pos_tags = ['x', 'y', 'z']
    pos = make_pos(hosts, pos_tags)
    gal_z = gals.zr.values
    host_z = hosts.zr.values
    dnbr = np.zeros(len(gals))
    res = [np.zeros(len(gals)) for attr in attrs]

    for i in xrange(len(gals)):
        if i % 10000 == 0: print i, len(gals)
        sel = np.where(np.abs(gal_z[i]- host_z) < width_by_2)[0]
        center = pos[i]
        with fast3tree(pos[sel]) as tree:
            r, ind = get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=nn, exclude_self=True)
            dnbr[i] = np.log10(r)
            for j, attr in enumerate(attrs):
                res[j][i] = hosts.ix[sel][attr].values[ind]
    return dnbr, res


def get_dist_and_attrs(hosts, gals, nn, attrs, box_size=250.0, projected=False):
    """
    hosts - parent set of halos
    gals - galaxy set
    nn - num neighbors
    attrs - list of attributes (i.e. ['mvir','vmax'])
    """
    if projected:
        pos_tags = ['x', 'y', 'zr']
    else:
        pos_tags = ['x', 'y', 'z']
    pos = make_pos(hosts, pos_tags)

    dnbr = np.zeros(len(gals))
    res = [np.zeros(len(gals)) for attr in attrs]
    with fast3tree(pos) as tree:
        for i in xrange(len(gals)):
            if i % 10000 == 0: print i, len(gals)
            center = [gals[tag].values[i] for tag in pos_tags]
            r, ind = get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=nn, exclude_self=True)
            dnbr[i] = np.log10(r)
            for j, attr in enumerate(attrs):
                res[j][i] = hosts[attr].values[ind]
    return dnbr, res


def wrap_boundary(pos, box_size):
    pos[pos < 0] += box_size
    pos[pos > box_size] -= box_size
    return pos

def get_distance(center, pos, box_size=-1):
    """
    Computes distance between points in 1D.

    Parameters
    ----------
    center : float
        central point
    pos : array-like
        other points computing distance to
    box_size : float (Delfault: -1)
        if box_size > 0, assumes periodic BCs

    Returns
    -------
    d : array-like
        distance between center and pos
    """
    d = pos - np.array(center)
    if box_size > 0:
        half_box_size = box_size*0.5
        d[d > half_box_size] -= box_size
        d[d < -half_box_size] += box_size
    return d


def get_3d_distance(center, pos, box_size=-1):
    dx = get_distance(center[0], pos[:, 0], box_size=box_size)
    dy = get_distance(center[1], pos[:, 1], box_size=box_size)
    dz = get_distance(center[2], pos[:, 2], box_size=box_size)
    r2 = dx*dx + dy*dy + dz*dz
    return np.sqrt(r2)


def get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=1,
                             exclude_self=True):
    """
    This function is massively inefficient. It makes two calls to the
    tree code because right now the tree code only returns the distance
    to the nearest neighbor, not the index or a pointer. The first call
    gets a fiducial distance in the primary simulation image. Then you
    find all points within that distance in all images and get the closest.
    Modified to be able to return the nth_nearest object specified by
    num_neighbors.
    """
    half_box_size = box_size/2.0
    tree.set_boundaries(0.0, box_size) ##!! important
    rfid = tree.query_nearest_distance(center)
    if rfid >= half_box_size:
        rfid = half_box_size - 2e-6
    if rfid == 0.0:
        rfid = box_size/np.power(tree.data.shape[0], 1.0/3.0)*10.0
        if rfid > half_box_size:
            rfid = half_box_size - 2e-6
    rfid += 1e-6
    double_count = 0
    while True:
        assert rfid < half_box_size
        idx, pos = tree.query_radius(center, rfid, periodic=box_size, output='both')
        if rfid == half_box_size - 1e-6:
            break
        if len(idx) < num_neighbors+1:
            rfid *= 2.0
            if rfid > half_box_size:
                rfid = half_box_size - 1e-6
        else:
            break
    r = get_3d_distance(center, pos, box_size)
    if exclude_self:
        msk = r > 0.0
        r = r[msk]
        idx = idx[msk]
    if num_neighbors < 0:
        q = np.argsort(r)
    elif len(r) < num_neighbors:
        return half_box_size, -1
    else:
        q = np.argsort(r)[num_neighbors - 1]
    return r[q], idx[q]

def calculate_xi(cat, box_size, projected=True, rpmin=0.1, rpmax=20, Nrp=25):
    """
    Given a catalog of galaxies, compute the correlation function using
    approriate helper functions from CorrelationFunction.py
    """
    rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
    pos = np.zeros((len(cat), 3), order='C')
    if projected:
        coords = ['x','y','zr']
    else:
        coords = ['x','y','z']
    for i, coord in enumerate(coords):
        pos[:,i] = cat[coord]/h
    # why nside=3?
    xi, cov, jack = projected_correlation(pos, rbins, zmax, box_size/h, jackknife_nside=3)
    return xi, cov, jack


def wprp_split(gals, red_split, box_size, cols=['ssfr','pred'],
                  rpmin=0.1, rpmax=20.0, Nrp=25): # for 2 splits
    # want the new format to be [ r, xra, xba, xrp, xbp, vr, vb]
    r, rbins = make_r_scale(rpmin, rpmax, Nrp)
    n_jack = 9      # hard coded value
    results = []
    results.append(r)
    r_jack = []
    b_jack = []
    for col in cols:
        red = gals[gals[col] < red_split]
        blue = gals[gals[col] > red_split]
        rx, rc, rj = calculate_xi(red, box_size, True, rpmin, rpmax, Nrp)
        bx, bc, bj = calculate_xi(blue, box_size, True, rpmin, rpmax, Nrp)
        results.append(rx)
        results.append(bx)
        r_jack.append(rj)
        b_jack.append(bj)
    r_var = np.sqrt(np.diag(np.cov(r_jack[0] - r_jack[1], rowvar=0, bias=1) * (n_jack -1)))
    b_var = np.sqrt(np.diag(np.cov(b_jack[0] - b_jack[1], rowvar=0, bias=1) * (n_jack -1)))
    results.append(r_var)
    results.append(b_var)
    return results


def wprp_bins(gals, num_splits, box_size, rpmin=0.1, rpmax=20.0, Nrp=25):
    """
    Takes in a data frame of galaxies and bins galaxies by ssfr. The CF is
    calculated for both predicted and actual ssfr values and passed to a helper
    function for plotting.
    """
    n_jack = 9      # hard coded value
    percentiles = [np.round(100. * i/(num_splits + 1)) for i in xrange(0, num_splits + 2)]
    bins = np.percentile(gals['ssfr'].values, percentiles)

    actual_dfs, pred_dfs = [], []
    for i in range(len(bins) - 1):
        actual_dfs.append(gals[(gals.ssfr > bins[i]) & (gals.ssfr < bins[i+1])])
        pred_dfs.append(gals[(gals.pred > bins[i]) & (gals.pred < bins[i+1])])

    r, rbins = make_r_scale(rpmin, rpmax, Nrp)
    results = [r]
    jacks = []
    for dfs in [actual_dfs, pred_dfs]:
        temp = []
        temp_jack = []
        for df in dfs:
            xi, cov, jack = calculate_xi(df, box_size, True, rpmin, rpmax, Nrp)
            temp.append(xi) # TODO: update this
            temp_jack.append(jack)
        jacks.append(temp_jack)
        results.append(temp)
    errs = []
    for i in xrange(num_splits + 1):
        errs.append(np.sqrt(np.diag(np.cov(jacks[0][i] - jacks[1][i], rowvar=0, bias=1)) * (n_jack - 1)))
    results.append(errs)

    return results  # r, ssfr, pred, errs


def find_min_rhill(rs, idxs, m_sec, larger_halos):
    if len(rs) <= 0:
        return [np.nan, np.nan, np.nan]
    else:
        rhills = [r * (m_sec/ (3 * larger_halos['mvir'].values[idx]) ) ** (1./3)
                  for r, idx in zip(rs, idxs)]
        rhill_min, idx = min([(r, i) for i, r in enumerate(rhills)])
        r_halo = rs[idx]
        m_halo = larger_halos['mvir'].values[idx]
        return rhill_min, r_halo, m_halo


def get_all_neighbors(pos, center, box_size):
    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        rmax = box_size/2 - 1e-6
        return tree.query_radius(center, rmax, periodic=box_size, output='both')


def calculate_r_hill(galaxies, hosts, box_size, projected=False):
    """
    calculate_r_hill iterates over all halos in the halo set. For each halo,
    calculate the rhill generated by all more massive halos save the r_hill min
    and corresponding distance and mass of the determining neighbor halo.
    """
    half_box_size = box_size/2
    hosts = hosts.reset_index(drop=True)
    galaxies = galaxies.reset_index(drop=True)
    r_hills = []
    halo_dists = []
    halo_masses = []

    if projected:
        pos_tags = ['x', 'y', 'zr']
    else:
        pos_tags = ['x', 'y', 'z']

    # Iterate over the halos and compare
    for i, galaxy in galaxies.iterrows():
        if i % 5000 == 0:
            print i
        m_sec = galaxy['mvir']
        center = [galaxy[tag] for tag in pos_tags]
        larger_halos = hosts[hosts['mvir'] > m_sec]
        pos = np.zeros((len(larger_halos), 3))
        for i, tag in enumerate(pos_tags):
            pos[:, i] = larger_halos[tag][:]
        num_tries = 0
        # Consider all massive neighbors
        idxs, loc = get_all_neighbors(pos, center, box_size)
        rs = get_3d_distance(center, loc, box_size)
        msk = rs > 0.0
        rs, idxs = rs[msk], idxs[msk]
        rhill, halo_dist, halo_mass = find_min_rhill(rs, idxs, m_sec, larger_halos)
        r_hills.append(rhill)
        halo_dists.append(halo_dist)
        halo_masses.append(halo_mass)

    return np.array(r_hills), np.array(halo_dists), np.array(halo_masses)


#def quenched_fraction(catalog, nbins=14):
#    masses, ssfr, pred = catalog.mvir.values, catalog.ssfr.values, catalog.pred.values
#    bins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nbins)
#
#    # TODO: needs error bars
#    red_cut = -11.0
#    actual_red, = np.where(ssfr < red_cut)
#    pred_red, = np.where(pred < red_cut)
#    actual_blue, = np.where(ssfr > red_cut)
#    pred_blue, = np.where(pred > red_cut)
#
#    actual_red_counts, _ = np.histogram(masses[actual_red], bins)
#    actual_blue_counts, _ = np.histogram(masses[actual_blue], bins)
#    pred_red_counts, _ = np.histogram(masses[pred_red], bins)
#    pred_blue_counts, _ = np.histogram(masses[pred_blue], bins)
#
#    f_q_actual = 1.0 * actual_red_counts / (actual_red_counts + actual_blue_counts)
#    f_q_predicted = 1.0 * pred_red_counts / (pred_red_counts + pred_blue_counts)
#
#    return f_q_actual, f_q_predicted


def color_counts_for_HOD(id_to_bin, objects, nbins, red_cut=-11.0, id='upid', col='ssfr'):
    blue_counts, red_counts = np.zeros(nbins), np.zeros(nbins)
    bins = pd.Series(objects[id]).map(id_to_bin)
    cols = pd.Series(objects[col]).map(lambda x: x < red_cut)
    for bin_id, red in zip(bins, cols):
        if not np.isnan(bin_id):
            if red:
                red_counts[bin_id] += 1
            else:
                blue_counts[bin_id] += 1
    return blue_counts, red_counts


def HOD(d0, test_gals, msmin=9.8, msmax=None, log_space=None):
    mvir = d0['mvir'].values
    red_cut = -11.0
    # create equal spacing on log scale
    if log_space is None:
        log_space = np.arange(np.log10(np.min(mvir)), np.log10(np.max(mvir)),.2)

    edges = 10**log_space
    centers = 10 ** ((log_space[1:] + log_space[:-1])/2)
    halos = d0[d0['upid'] == -1]
    centrals = halos[halos['mstar'] > msmin]
    satellites = d0[(d0['upid'] != -1) & (d0['mstar'] > msmin)]
    if msmax:
        satellites = satellites[satellites['mstar'] < msmax]
        centrals = centrals[centrals['mstar'] < msmax]

    # count the number of parents in each bin
    num_halos, _ = np.histogram(halos['mvir'], edges)
    num_halos = num_halos.astype(np.float)
    nbins = len(centers)

    # create map from upid to host mass to bin
    halo_id_to_bin = {}
    for _, halo in halos.iterrows():
        bin_id = np.digitize([halo['mvir']], edges, right=True)[0]
        halo_id_to_bin[halo['id']] = min(bin_id, nbins-1)

    num_actual_blue_s, num_actual_red_s = color_counts_for_HOD(halo_id_to_bin, satellites, nbins, id='upid', col='ssfr')
    num_actual_blue_c, num_actual_red_c = color_counts_for_HOD(halo_id_to_bin, centrals, nbins, id='id', col='ssfr')

    pred_sats = test_gals[test_gals['upid'] != -1]
    pred_cents = test_gals[test_gals['upid'] == -1]
    num_pred_blue_s, num_pred_red_s = color_counts_for_HOD(halo_id_to_bin, pred_sats, nbins, id='upid', col='pred')
    num_pred_blue_c, num_pred_red_c = color_counts_for_HOD(halo_id_to_bin, pred_cents, nbins, id='id', col='pred')
    # TODO: verify that I should be comparing the real color of the full thing to 8/7 times the predicted ones
    results = []
    results.append(centers)
    results.append([[num_actual_red_c/num_halos, num_actual_blue_c/num_halos], [num_actual_red_s/num_halos, num_actual_blue_s/num_halos]])
    results.append([[num_pred_red_c/num_halos, num_pred_blue_c/num_halos], [num_pred_red_s/num_halos, num_pred_blue_s/num_halos]])
    return results


def HOD_wrapper(df, test_gals, box_size):
    full_octants = util.jackknife_octant_samples(df, box_size)
    test_octants = util.jackknife_octant_samples(test_gals, box_size)
    log_space = np.arange(np.log10(np.min(df.mvir)), np.log10(np.max(df.mvir)),.2)
    centers = 10 ** ((log_space[1:] + log_space[:-1])/2)
    oct_hods = []
    for full, test in zip(full_octants, test_octants):
        oct_hods.append(HOD(full, test, log_space=log_space))
    red_c_a = np.array([result[1][0][0] for result in oct_hods])
    blue_c_a = np.array([result[1][0][1] for result in oct_hods])
    red_s_a = np.array([result[1][1][0] for result in oct_hods])
    blue_s_a = np.array([result[1][1][1] for result in oct_hods])
    red_c_p = np.array([result[2][0][0] for result in oct_hods])
    blue_c_p = np.array([result[2][0][1] for result in oct_hods])
    red_s_p = np.array([result[2][1][0] for result in oct_hods])
    blue_s_p = np.array([result[2][1][1] for result in oct_hods])
    # two options
    # option 1: take the variance to be the mean std of true-pred
    # option 2: take jackknife variance independently
    # should these be taken over multiple runs of the color assignment?

    # results structure [centers, [total], [sf/q] , [c], [s]]
    n_jack = len(oct_hods)
    results = [centers]
    totals_a = [rc + bc + rs + bs for rc,bc,rs,bs in zip(red_c_a, blue_c_a, red_s_a, blue_s_a)]
    totals_p = [rc + bc + rs + bs for rc,bc,rs,bs in zip(red_c_p, blue_c_p, red_s_p, blue_s_p)]
    results.append([np.mean(totals_a, axis=0), np.mean(totals_p, axis=0),
        np.sqrt(np.diag(np.cov(np.array(totals_a)-np.array(totals_p), rowvar=0, bias=1)) * (n_jack - 1))])

    red_a = [rc + rs for rc, rs in zip(red_c_a, red_s_a)]
    blue_a = [bc + bs for bc, bs in zip(blue_c_a, blue_s_a)]
    red_p = [rc + rs for rc, rs in zip(red_c_p, red_s_p)]
    blue_p = [bc + bs for bc, bs in zip(blue_c_p, blue_s_p)]

    results.append([np.mean(red_a, axis=0), np.mean(blue_a, axis=0),
        np.mean(red_p, axis=0), np.mean(blue_p, axis=0),
        np.sqrt(np.diag(np.cov(np.array(red_a) - np.array(red_p), rowvar=0, bias=1)) * (n_jack -1)),
        np.sqrt(np.diag(np.cov(np.array(blue_a) - np.array(blue_p), rowvar=0, bias=1)) * (n_jack -1))])

    results.append([np.mean(red_c_a, axis=0), np.mean(blue_c_a, axis=0),
        np.mean(red_c_p, axis=0), np.mean(blue_c_p, axis=0),
        np.sqrt(np.diag(np.cov(np.array(red_c_a) - np.array(red_c_p), rowvar=0, bias=1)) * (n_jack -1)),
        np.sqrt(np.diag(np.cov(np.array(blue_c_a) - np.array(blue_c_p), rowvar=0, bias=1)) * (n_jack -1))])

    results.append([np.mean(red_s_a, axis=0), np.mean(blue_s_a, axis=0),
        np.mean(red_s_p, axis=0), np.mean(blue_s_p, axis=0),
        np.sqrt(np.diag(np.cov(np.array(red_s_a) - np.array(red_s_p), rowvar=0, bias=1)) * (n_jack -1)),
        np.sqrt(np.diag(np.cov(np.array(blue_s_a) - np.array(blue_s_p), rowvar=0, bias=1)) * (n_jack -1))])

    return results


def radial_profile_counts(gals, hosts, box_size, r, rbins, rmax, col='ssfr'):
    num_halos = len(hosts)
    results = []
    pos = make_pos(gals)
    pos_tags = ['x', 'y', 'zr']
    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        #mass_select = hosts[hosts['mbin_idx'] == i]
        num_reds, num_blues = [],[]
        num_pred_reds, num_pred_blues = [], []
        diff_reds, diff_blues = [], []
        for j, halo in hosts.iterrows():
            num_red, num_blue = np.zeros(len(r)), np.zeros(len(r))
            num_pred_red, num_pred_blue = np.zeros(len(r)), np.zeros(len(r))
            center = [halo[tag] for tag in pos_tags]
            idxs, pos = tree.query_radius(center, rmax, periodic=box_size, output='both')
            rs = get_3d_distance(center, pos, box_size=box_size)
            msk = rs > 0
            rs = rs[msk]
            idxs = idxs[msk]
            for dist, sat_idx in zip(rs, idxs):
                rbin = np.digitize([dist], rbins) - 1 # -1 for the r vs rbin
                # indexing reflects return values from tree
                if gals['ssfr'].values[sat_idx] < -11:
                    num_red[rbin] += 1
                else:
                    num_blue[rbin] += 1
                if gals['pred'].values[sat_idx] < -11:
                    num_pred_red[rbin] += 1
                else:
                    num_pred_blue[rbin] += 1
            num_reds.append(num_red.cumsum())
            num_blues.append(num_blue.cumsum())
            num_pred_reds.append(num_pred_red.cumsum())
            num_pred_blues.append(num_pred_blue.cumsum())
            diff_reds.append(num_red.cumsum() - num_pred_red.cumsum())
            diff_blues.append(num_blue.cumsum() - num_pred_blue.cumsum())
    all_counts = map(np.array, [num_reds, num_blues, num_pred_reds, num_pred_blues])
    means = map(lambda x: np.mean(x, axis=0), all_counts)
    #print np.mean(num_pred_reds-means[0], axis=0);
    #print np.std(num_pred_reds-means[0], axis=0);
    #stds = map(lambda x: np.std(x, axis=0), [num_pred_reds-means[0], num_pred_blues-means[1]]) #[diff_reds, diff_blues])
    stds = map(lambda x: np.std(x, axis=0), [num_reds-means[0], num_blues-means[1]]) #[diff_reds, diff_blues])
    #stds = map(lambda x: np.std(x, axis=0), [num_reds, num_blues]) #[diff_reds, diff_blues])
    #stds = map(lambda x: np.std(x, axis=0), [np.array(diff_reds), np.array(diff_blues)])
    volumes = [4./3 * np.pi * rad**3 for rad in r] # maybe rbins
    # Do these need any cumsums? # don't think so
    for counts in means:
        results.append(counts / volumes)
    for errs in stds:
        results.append(errs / volumes)
    return results

def radial_profile(hosts, gals, box_size, rmin=0.1, rmax=5.0, Nrp=10):
    r, rbins = make_r_scale(rmin, rmax, Nrp)
    results = [r]
    gals = gals[gals.upid != -1]
    gals = gals.reset_index(drop=True)

    mvir = hosts['mvir'].values
    delta = 0.25
    masses = [12,13,14]
    mlims = [[10**(mass-delta), 10**(mass+delta)] for mass in masses]

    for lim in mlims:
        host_sel = hosts[(hosts.mvir > lim[0]) & (hosts.mvir < lim[1])]
        host_ids = set(host_sel.id.values)
        gal_sel = gals.copy()
        gal_sel['include'] = [upid in host_ids for upid in gal_sel.upid.values]
        gal_sel = gal_sel[gal_sel['include'] == True]
        host_sel = host_sel.reset_index(drop=True)
        gal_sel = gal_sel.reset_index(drop=True)
        counts = radial_profile_counts(gal_sel, host_sel, box_size, r, rbins, rmax)
        results.append(counts)

    return results


def density_profile_wrapper(hosts, gals, box_size, rmin=0.1, rmax=5.0, Nrp=10):
    hosts = hosts.reset_index(drop=True)
    octants = util.jackknife_octant_samples(gals, box_size)
    host_octs = util.jackknife_octant_samples(hosts, box_size)
    r, rbins = make_r_scale(rmin, rmax, Nrp)
    results = [r]

    profiles = []
    for i, (octant, host_oct) in enumerate(zip(octants, host_octs)):
        #print 'octant: ', i
        profiles.append(density_profile(host_oct, octant, box_size, rmin, rmax, Nrp))

    n_jack = len(octants)
    #print len(profiles[0][2])
    for i in xrange(3):
        num_reds = np.array([result[i+1][0] for result in profiles])
        num_blues = np.array([result[i+1][1] for result in profiles])
        num_pred_reds = np.array([result[i+1][2] for result in profiles])
        num_pred_blues = np.array([result[i+1][3] for result in profiles])
        results.append([np.mean(num_reds, axis=0), np.mean(num_blues, axis=0),
            np.mean(num_pred_reds, axis=0), np.mean(num_pred_blues, axis=0),
            np.sqrt(np.diag(np.cov(num_reds - num_pred_reds, rowvar=0, bias=1)) * (n_jack - 1)),
            np.sqrt(np.diag(np.cov(num_blues - num_pred_blues, rowvar=0, bias=1)) * (n_jack - 1))])

    return results


def counts_for_fq(df, red_cut, col, x_vals, bins):
    sel = np.where(df[col])
    red_sel = np.where(df[col] < red_cut)[0]
    blue_sel = np.where(df[col] > red_cut)[0]
    red_counts,_ = np.histogram(x_vals[red_sel], bins)
    blue_counts,_ = np.histogram(x_vals[blue_sel], bins)
    return [red_counts.astype(float), blue_counts.astype(float)]


def quenched_fraction(gals, red_cut, host_mass_dict, mbins=None, nbins=14):
    cents = gals[gals.upid == -1]
    sats = gals[gals.upid != -1]
    sats.reset_index(drop=True)
    cents = cents.reset_index(drop=True)
    central_masses = cents.mvir.values
    host_ids = set(host_mass_dict.keys())
    sats = sats[sats.upid.isin(host_ids)]
    sats = sats.reset_index(drop=True)
    sat_host_masses = np.array([host_mass_dict[sat.upid] for _, sat in sats.iterrows()])

    if mbins is None:
        masses = gals.mvir.values
        mbins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nbins)
    centers = np.sqrt(mbins[:-1] * mbins[1:])
    results = [centers]

    for col in ['ssfr', 'pred']:
        temp = []
        temp.append(counts_for_fq(cents, red_cut, col, central_masses, mbins))
        temp.append(counts_for_fq(sats, red_cut, col, sat_host_masses, mbins))
        results.append(temp)

    return results # centers, [[a_c_r, a_c_b], a_s], [p_c, p_s]


def quenched_fraction_wrapper(gals, box_size, mass_dict, red_cut=-11.0, nbins=14):
    octants = util.jackknife_octant_samples(gals, box_size)
    masses = gals.mvir.values
    mbins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nbins)
    centers = np.sqrt(mbins[:-1] * mbins[1:])
    fq_s = []
    for octant in octants:
        fq_s.append(quenched_fraction(octant, red_cut, mass_dict, mbins))
    red_c_a = np.array([result[1][0][0] for result in fq_s])
    blue_c_a = np.array([result[1][0][1] for result in fq_s])
    red_s_a = np.array([result[1][1][0] for result in fq_s])
    blue_s_a = np.array([result[1][1][1] for result in fq_s])
    red_c_p = np.array([result[2][0][0] for result in fq_s])
    blue_c_p = np.array([result[2][0][1] for result in fq_s])
    red_s_p = np.array([result[2][1][0] for result in fq_s])
    blue_s_p = np.array([result[2][1][1] for result in fq_s])

    n_jack = len(octants)
    results = [centers] # then fq_total fq_cent fq_sat + corresponding errors
    actual_fqs = red_c_a/(red_c_a + blue_c_a)
    pred_fqs = red_c_p/(red_c_p + blue_c_p)
    fq_total_actual = np.mean(actual_fqs,  axis=0)
    fq_total_pred = np.mean(pred_fqs,  axis=0)
    fq_total_err = np.sqrt(np.diag(np.cov(actual_fqs - pred_fqs, rowvar=0, bias=1)) * (n_jack - 1))
    results.append([fq_total_actual, fq_total_pred, fq_total_err])

    actual_fqs = red_s_a/(red_s_a + blue_s_a)
    pred_fqs = red_s_p/(red_s_p + blue_s_p)
    fq_total_actual = np.mean(actual_fqs,  axis=0)
    fq_total_pred = np.mean(pred_fqs,  axis=0)
    fq_total_err = np.sqrt(np.diag(np.cov(actual_fqs - pred_fqs, rowvar=0, bias=1)) * (n_jack - 1))
    results.append([fq_total_actual, fq_total_pred, fq_total_err])

    actual_fqs = (red_c_a + red_s_a)/(red_c_a + blue_c_a + red_s_a + blue_s_a)
    pred_fqs = (red_c_p + red_s_p)/(red_c_p + blue_c_p + red_s_p + blue_s_p)
    fq_total_actual = np.mean(actual_fqs,  axis=0)
    fq_total_pred = np.mean(pred_fqs,  axis=0)
    fq_total_err = np.sqrt(np.diag(np.cov(actual_fqs - pred_fqs, rowvar=0, bias=1)) * (n_jack - 1))
    results.append([fq_total_actual, fq_total_pred, fq_total_err])

    return results



def density_vs_fq(gals, cutoffs, red_cut=-11, dbins=None, nbins=10):
    sections = [gals[(gals['mstar'] >= cutoffs[i]) & (gals['mstar'] < cutoffs[i+1])] for i in xrange(len(cutoffs)-1)]
    s5 = gals['$\Sigma_{5}$'].values
    print len(s5)
    if dbins is None:
        dbins = np.linspace(min(s5), max(s5), nbins)
    centers = (dbins[:-1] + dbins[1:])/2 # average because we're already in logspace
    results = [cutoffs, centers]

    for col in ['ssfr', 'pred']:
        temp = []
        for section in sections:
            dists = section['$\Sigma_{5}$'].values
            temp.append(counts_for_fq(section, red_cut, col, dists, dbins))
        results.append(temp)
    return results


def density_vs_fq_wrapper(gals, box_size, cutoffs, red_cut=-11.0, nbins=10):
    octants = util.jackknife_octant_samples(gals, box_size)
    s5 = gals['$\Sigma_{5}$'].values
    dbins = np.linspace(min(s5), max(s5), nbins)
    centers = (dbins[:-1] + dbins[1:])/2
    fq_s = []
    for octant in octants:
        fq_s.append(density_vs_fq(octant, cutoffs, red_cut=red_cut, dbins=dbins, nbins=nbins))

    n_jack = len(octants)
    actual_fqs, pred_fqs, errs = [], [], []
    for i in xrange(len(cutoffs)-1):
        actuals = np.array([(result[2][i][0]) / (result[2][i][0] + result[2][i][1]) for result in fq_s])
        preds = np.array([(result[3][i][0]) / (result[3][i][0] + result[3][i][1]) for result in fq_s])
        actual_fqs.append(np.mean(actuals, axis=0))
        pred_fqs.append(np.mean(preds, axis=0))
        errs.append(np.sqrt(np.diag(np.cov(actuals - preds, rowvar=0, bias=1)) * (n_jack - 1)))

    results = [cutoffs, centers, actual_fqs, pred_fqs, errs]
    return results


def density_match(gals, box_size, HW, sm_cuts, debug=False):
    # sample_size = 1./box_size # I think this is how I came up with the number
    densities = []
    test_cuts = np.arange(9,11.5,.01)
    for cut in test_cuts:
        #densities.append(len(gals[gals.mstar > cut]) / box_size**3/sample_size)
        densities.append(len(gals[gals.mstar > cut]) / box_size**3)

    actual_densities = [len(HW[HW.mstar > cut])/250.0**3 for cut in sm_cuts]

    if debug:
        print 'HW densities are: ',actual_densities
        plt.plot(test_cuts, densities, '.')
        plt.plot(sm_cuts, actual_densities, '*')
        p.style_plots()

    # matching
    idxs = np.digitize(actual_densities, densities)
    return test_cuts[idxs]


def match_quenched_fraction(gals, new_sm_cuts, HW, sm_cuts, red_cut, debug=False):
    def fq_helper(gals, cut):
        return 1.0 * sum(gals.ssfr < cut) / len(gals)
    actual_f_q = [fq_helper(HW[HW.mstar > cut], red_cut) for cut in sm_cuts]
    print actual_f_q
    new_red_cuts = []
    test_red_cuts = np.arange(-11.5,-9.5,.01)
    if debug:
        sns.kdeplot(gals.ssfr)
        plt.figure()
    for i, (new_sm_cut, old_sm_cut) in enumerate(zip(new_sm_cuts, sm_cuts)):
        quenched_fractions = []
        for cut in test_red_cuts:
            quenched_fractions.append(fq_helper(gals[gals.mstar > new_sm_cut], cut))
        if debug:
            plt.plot(test_red_cuts, quenched_fractions)
        idx = np.digitize([actual_f_q[i]], quenched_fractions)[0] # array and dearray
        new_red_cuts.append(test_red_cuts[idx])
    if debug:
        plt.plot([red_cut] * 3, actual_f_q, '*')
        p.style_plots()
    return new_red_cuts


def abundance_match(gals, box_size, debug=False):
    '''
    Returns the stellar mass cuts which match the density of the H&W catalog
    along with the cuts in sSFR to make the quenched fraction correct at each
    of the stellar mass cuts.
    '''
    # TODO: Make sure that all the catalogs are using units of log mstar

    # Hard coded values to match against
    d_hw = util.fits_to_pandas(fitsio.read('data/HW/Becker_CAM_mock.fits', lower=True))
    sm_cuts = [9.8, 10.2, 10.6]
    red_cut = -11

    if debug:
        plt.figure()
    new_sm_cuts = density_match(gals, box_size, d_hw, sm_cuts, debug)

    if debug:
        plt.figure()
        for cut in new_sm_cuts:
            sns.kdeplot(gals[gals.mstar > cut].ssfr)

    if debug:
        plt.figure()
    red_cuts = match_quenched_fraction(gals, new_sm_cuts, d_hw, sm_cuts, red_cut, debug)
    print new_sm_cuts
    print red_cuts
    return new_sm_cuts, red_cuts


def calculate_projected_z(df):
    c = 3e5     # km/s
    table = generate_z_of_r_table(0.3, 0.7)
    zred = z_of_r(df['z'], table)
    df['zr'] = zred + df['vz']/c
    return df


def generate_z_of_r_table(omegam, omegal, zmax=2.0, npts=1000):

    c = 2.9979e5
    da = (1.0 - (1.0/(zmax+1)))/npts
    dtype = np.dtype([('r', np.float), ('z', np.float)])
    z_of_r_table = np.ndarray(npts, dtype=dtype)
    Thisa = 1.0
    z_of_r_table['z'][0] = 1.0/Thisa - 1.
    z_of_r_table['r'][0] = 0.0
    for i in range(1, npts):
        Thisa = 1. - da*float(i)
        ThisH = 100.*np.sqrt(omegam/Thisa**3 + omegal)
        z_of_r_table['z'][i] = 1./Thisa - 1
        z_of_r_table['r'][i] = z_of_r_table['r'][i-1] + 1./(ThisH*Thisa*Thisa)*da*c
    return z_of_r_table

def z_of_r(r, table):

    npts = len(table)
    try:
        nz = len(r)
    except:
        nz = 1

    zred = np.zeros(nz)-1

    if nz==1:
        for i in range(1, npts):
            if (table['r'][i] > r): break
        slope = (table['z'][i] - table['z'][i-1])/(table['r'][i]-table['r'][i-1])
        zred = table['z'][i-1] + slope*(r-table['r'][i-1])
    else:
        for i in range(1, npts):
            ii, = np.where((r >= table['r'][i-1]) & (r < table['r'][i]))
            count = len(ii)
            if (count == 0): continue
            slope = (table['z'][i] - table['z'][i-1])/(table['r'][i]-table['r'][i-1])
            zred[ii] = table['z'][i-1] + slope*(r[ii]-table['r'][i-1])

    return zred

def ssfr_from_sfr(x):
    if x.sfr == 0.0:
        return -12
    else:
        return np.log10(x.sfr/(10**x.mstar))

def color_am(ssfr, rhill):
    r_order = np.argsort(rhill)
    ssfr_sorted = np.array(sorted(ssfr))
    pred_colors = np.zeros(len(ssfr))
    pred_colors[r_order] = ssfr_sorted
    return pred_colors
