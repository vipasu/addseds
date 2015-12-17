from CorrelationFunction import projected_correlation
import os
import errno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fast3tree import fast3tree

### TODO: define functionality
# API Calls
# c.wprp
# c.wprp_split
# c.HOD
# c.density_profile

h = 0.7
L = 250.0/h
zmax = 40.0

rpmin = 0.1
rpmax = 20.0
Nrp = 25
rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
r = np.sqrt(rbins[1:]*rbins[:-1])


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


def get_dist_and_attrs(hosts, gals, nn, attrs, box_size=250.0, projected=False):
    """
    hosts - parent set of halos
    gals - galaxy set
    nn - num neighbors
    attrs - list of attributes (i.e. ['mvir','vmax'])
    """
    pos = np.zeros((len(hosts), 3))
    if projected:
        pos_tags = ['x', 'y', 'zr']
    else:
        pos_tags = ['x', 'y', 'z']
    for i, tag in enumerate(pos_tags):
        pos[:, i] = hosts[tag][:]

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
    dx = get_distance(center[0], loc[:, 0], box_size=box_size)
    dy = get_distance(center[1], loc[:, 1], box_size=box_size)
    dz = get_distance(center[2], loc[:, 2], box_size=box_size)
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
    if rfid == 0.0:
        rfid = box_size/np.power(tree.data.shape[0], 1.0/3.0)*10.0
        if rfid > half_box_size:
            rfid = half_box_size - 2e-6
    rfid += 1e-6
    while True:
        assert rfid < half_box_size
        idx, pos = tree.query_radius(center, rfid, periodic=box_size, output='both')
        # if len(idx) <= 1:
        if len(idx) <= num_neighbors:
            rfid *= 2.0
        else:
            break
    dx = get_distance(center[0], pos[:, 0], box_size=box_size)
    dy = get_distance(center[1], pos[:, 1], box_size=box_size)
    dz = get_distance(center[2], pos[:, 2], box_size=box_size)
    r2 = dx*dx + dy*dy + dz*dz
    if exclude_self:
        msk = r2 > 0.0
        r2 = r2[msk]
        idx = idx[msk]
    if num_neighbors < 0:
        q = np.argsort(r2)
    else:
        q = np.argsort(r2)[num_neighbors - 1]
    return np.sqrt(r2[q]), idx[q]

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
    xi, cov = projected_correlation(pos, rbins, zmax, box_size/h, jackknife_nside=3)
    return xi, cov


def wprp_split(gals, red_split, box_size, cols=['ssfr','pred'],
                  rpmin=0.1, rpmax=20.0, Nrp=25): # for 2 splits
    results = []
    for col in cols:
        red = gals[gals[col] < red_split]
        blue = gals[gals[col] > red_split]
        rx, rc = calculate_xi(red, box_size, True, rpmin, rpmax, Nrp)
        bx, bc = calculate_xi(blue, box_size, True, rpmin, rpmax, Nrp)
        red = [rx, np.sqrt(np.diag(rc))]
        blue = [bx, np.sqrt(np.diag(bc))]
        results.append([red, blue])
    return results


def wprp_bins(gals, num_splits, box_size, rpmin=0.1, rpmax=20.0, Nrp=25):
    """
    Takes in a data frame of galaxies and bins galaxies by ssfr. The CF is
    calculated for both predicted and actual ssfr values and passed to a helper
    function for plotting.
    """
    percentiles = [np.round(100. * i/(num_splits + 1)) for i in xrange(0, num_splits + 2)]
    bins = np.percentile(gals['ssfr'].values, percentiles)

    actual_dfs, pred_dfs = [], []
    for i in range(len(bins) - 1):
        actual_dfs.append(gals[(gals.ssfr > bins[i]) & (gals.ssfr < bins[i+1])])
        pred_dfs.append(gals[(gals.pred > bins[i]) & (gals.pred < bins[i+1])])

    results = []
    for dfs in [actual_dfs, pred_dfs]:
        temp = []
        for df in dfs:
            xi, cov = calculate_xi(df, box_size, True, rpmin, rpmax, Nrp)
            temp.append([xi, np.sqrt(np.diag(cov))])
        results.append(temp)

    return results  # ssfr, pred


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
        idxs = idxs[msk]
        rhill, halo_dist, halo_mass = find_min_rhill(rs, idxs, m_sec, larger_halos)
        r_hills.append(rhill)
        halo_dists.append(halo_dist)
        halo_masses.append(halo_mass)

    return np.array(r_hills), np.array(halo_dists), np.array(halo_masses)


def quenched_fraction(catalog, nbins=14):
    masses, ssfr, pred = catalog.mvir.values, catalog.ssfr.values, catalog.pred.values
    bins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nbins)

    # TODO: needs error bars
    red_cut = -11.0
    actual_red, = np.where(ssfr < red_cut)
    pred_red, = np.where(pred < red_cut)
    actual_blue, = np.where(ssfr > red_cut)
    pred_blue, = np.where(pred > red_cut)

    actual_red_counts, _ = np.histogram(masses[actual_red], bins)
    actual_blue_counts, _ = np.histogram(masses[actual_blue], bins)
    pred_red_counts, _ = np.histogram(masses[pred_red], bins)
    pred_blue_counts, _ = np.histogram(masses[pred_blue], bins)

    f_q_actual = 1.0 * actual_red_counts / (actual_red_counts + actual_blue_counts)
    f_q_predicted = 1.0 * pred_red_counts / (pred_red_counts + pred_blue_counts)

    return f_q_actual, f_q_predicted


def HOD():
    pass


def density_profile():
    pass


def density_vs_fq():
    pass



def calculate_projected_z(df):
    df['zr'] = df['z'] + df['vz']/100 # TODO: is this the right unit conversion?
    return df


