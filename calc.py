"""
calc.py contains routines that are core to nearest neighbor, rhill, and halo
statistic calculations.
"""

from collections import Counter, defaultdict
from CorrelationFunction import projected_correlation
import pandas as pd
import numpy as np
from numpy.linalg import pinv, inv
from fast3tree import fast3tree
import util
from scipy.stats import rankdata

h = 0.7
zmax = 40.0

rpmin = 0.1
rpmax = 20.0
Nrp = 25
rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
r = np.sqrt(rbins[1:]*rbins[:-1])


def make_pos(gals, pos_tags=['x', 'y', 'z']):
    """ Makes an array of of shape (N,3) with specified coordinates."""
    pos = np.zeros((len(gals[pos_tags[0]]), 3))
    for i, tag in enumerate(pos_tags):
        pos[:, i] = gals[tag][:]
    return pos


def make_r_scale(rmin, rmax, Nrp):
    """ Provides bins and their centers between rmin and rmax."""
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), Nrp+1)
    r = np.sqrt(rbins[1:]*rbins[:-1])
    return r, rbins


def get_projected_dist_and_attrs(hosts, gals, nn, attrs, box_size=250.0):
    """
    Selects the nn-th nearest neighbor in redshift space from gals in hosts.

    Accepts:
        hosts - list of objects to search for nearest neighbors
        gals - objects to find nearest neighbors of
        nn - specifies which nearest neighbor to find
        attrs - list of properties to grab from halos (i.e. ['mvir', 'c'])
    Returns:
        dnbr - distances to the nn-th neighbor
        res - array of shape (len(attrs), len(gals))
    """
    width_by_2 = 0.01
    pos_tags = ['x', 'y']
    host_pos = make_pos(hosts, pos_tags)
    gal_pos = make_pos(gals, pos_tags)
    N = len(gals[pos_tags[0]])
    gal_z = gals['zr']
    host_z = hosts['zr']
    dnbr = np.zeros(N)
    res = [np.zeros(N) for attr in attrs]

    for i in xrange(N):
        if i % 10000 == 0:
            print i, N
        sel = np.where(np.abs(gal_z[i] - host_z) < width_by_2)[0]
        center = gal_pos[i]
        with fast3tree(host_pos[sel]) as tree:
            if len(sel) <= nn:
                print "Insufficient number of neighbors in redshift bin"
                print "redshift: ", gal_z[i]
                assert False
            r, ind = get_nearest_nbr_periodic(center, tree, box_size,
                                              num_neighbors=nn,
                                              exclude_self=True)
            dnbr[i] = np.log10(r)
            for j, attr in enumerate(attrs):
                res[j][i] = hosts[attr][sel][ind]
    return dnbr, res


def get_dist_and_attrs(hosts, gals, nn, attrs, box_size=250.0):
    """
    Selects the nn-th nearest neighbor in real space from gals in hosts.

    Accepts:
        hosts - list of objects to search for nearest neighbors
        gals - objects to find nearest neighbors of
        nn - specifies which nearest neighbor to find
        attrs - list of properties to grab from halos (i.e. ['mvir', 'c'])
    Returns:
        dnbr - distances to the nn-th neighbor
        res - array of shape (len(attrs), len(gals))
    """
    pos_tags = ['x', 'y', 'z']
    N = len(gals[pos_tags[0]])
    pos = make_pos(hosts, pos_tags)

    dnbr = np.zeros(N)
    res = [np.zeros(N) for attr in attrs]
    with fast3tree(pos) as tree:
        for i in xrange(N):
            if i % 10000 == 0:
                print i, N
            center = [gals[tag][i] for tag in pos_tags]
            r, ind = get_nearest_nbr_periodic(center, tree, box_size,
                                              num_neighbors=nn,
                                              exclude_self=True)
            dnbr[i] = np.log10(r)
            for j, attr in enumerate(attrs):
                res[j][i] = hosts[attr][ind]
    return dnbr, res


def wrap_boundary(pos, box_size):
    """ Enforces that values in pos fall between 0 and box_size. """
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


def get_cylinder_distance(center, pos, box_size=-1, zmax=40.0):
    """
    Computes distance between points in xy plane. If the z distance is greater
    than zmax, then box_size is reported as the distance.
    """
    dx = get_distance(center[0], pos[:, 0], box_size=box_size)
    dy = get_distance(center[1], pos[:, 1], box_size=box_size)
    dz = get_distance(center[2], pos[:, 2], box_size=box_size)
    r2 = dx*dx + dy*dy
    outside = np.where(np.abs(dz) > zmax)
    r2[outside] = box_size
    return np.sqrt(r2)


def get_3d_distance(center, pos, box_size=-1):
    """
    Computes distance between points in 3D.
    """
    dx = get_distance(center[0], pos[:, 0], box_size=box_size)
    dy = get_distance(center[1], pos[:, 1], box_size=box_size)
    dz = get_distance(center[2], pos[:, 2], box_size=box_size)
    r2 = dx*dx + dy*dy + dz*dz
    return np.sqrt(r2)


def get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=1):
    """
    Locates the num_neighbors-th nearest object from center in tree.
    Assumes periodic boundary conditions.
    The query radius around center is iteratively doubled until there are at
    least num_neighbors objects within the query radius.

    Accepts:
        center - 3d coordinate for query point
        tree - fast3tree containing neighbors to search
        box_size - size at which to wrap around coordinates
        num_neighbors - which nearest neighbor to return

    Returns:
        r - distance to the num_neighbors nearest neighbor
        idx - index in the tree corresponding to the object found
    """
    half_box_size = box_size/2.0
    tree.set_boundaries(0.0, box_size)
    rfid = tree.query_nearest_distance(center)
    if rfid >= half_box_size:
        rfid = half_box_size - 2e-6
    if rfid == 0.0:
        rfid = box_size/np.power(tree.data.shape[0], 1.0/3.0)*10.0
        if rfid > half_box_size:
            rfid = half_box_size - 2e-6
    rfid += 1e-6
    while True:
        assert rfid < half_box_size
        idx, pos = tree.query_radius(center, rfid, periodic=box_size,
                                     output='both')
        if rfid == half_box_size - 1e-6:
            break
        if len(idx) < num_neighbors+1:
            rfid *= 2.0
            if rfid > half_box_size:
                rfid = half_box_size - 1e-6
        else:
            break
    r = get_3d_distance(center, pos, box_size)
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


def calculate_xi(gals, box_size, projected=True, jack_nside=3, rpmin=0.1,
                 rpmax=20, Nrp=25):
    """
    Given a catalog of galaxies, compute the correlation function using
    approriate helper functions from CorrelationFunction.py
    """
    rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
    pos = np.zeros((len(gals), 3), order='C')
    if projected:
        coords = ['x', 'y', 'zp']
    else:
        coords = ['x', 'y', 'z']
    for i, coord in enumerate(coords):
        pos[:, i] = gals[coord]/h
    return projected_correlation(pos, rbins, zmax, box_size/h,
                                 jackknife_nside=jack_nside)


def calculate_chi_square(truth, pred, Sigma, Sigma_inv=None):
    """
    Calculates the chi squared value from two distributions and the covariance.

    Accepts:
    truth, pred - array-like objects to compare the goodness of fit
    Sigma - Covariance matrix (usually from jackknife)
    Sigma_inv - Optionally pass in the inverted covariance matrix.
    """
    if Sigma_inv is None:
        try:
            Sigma_inv = pinv(Sigma)
        except:
            Sigma_inv = inv(Sigma)
    d = truth - pred
    return np.dot(d, np.dot(Sigma_inv, d))/(len(d) - 1)


def wprp_split(gals, red_split, box_size, cols=['ssfr', 'pred'], jack_nside=3,
               rpmin=0.1, rpmax=20.0, Nrp=25):  # for 2 splits
    """
    Calculates the 2PCF of gals binned by sSFR, separated by red_split.

    Note that sSFR can be substitued in _cols_ to bin by, say, concentration

    Accepts:
        gals - numpy array with objects, their positions, and attributes
        red_split - value which separates two populations
        box_size - box_size of the objects in gals
        cols - tags to specify the actual and predicted distribution. Defaults
               to ['ssfr', 'pred'], but could be modified to use, say
               ['c', 'pred_c'] (assuming they exist in gals).

    Returns:
        [r, [actual], [pred], [err], [chi2]]
            r - centers of r bins
            [actual] - clustering of red/blue galaxies
            [pred] - clustering of predicted red/blue galaxies
            [err] - errorbars for red/blue galaxies
            [chi2] - goodness of fit for red/blue galaxies
    """
    r, rbins = make_r_scale(rpmin, rpmax, Nrp)
    n_jack = jack_nside ** 2
    results = []
    results.append(r)
    r_jack = []
    b_jack = []
    for col in cols:
        red = gals[gals[col] < red_split]
        blue = gals[gals[col] > red_split]
        r = calculate_xi(red, box_size, True, jack_nside, rpmin, rpmax, Nrp)
        b = calculate_xi(blue, box_size, True, jack_nside, rpmin, rpmax, Nrp)
        results.append([r[0], b[0]])
        if jack_nside <= 1:
            r_var = r[1]
            b_var = b[1]
        else:
            r_jack.append(r[2])
            b_jack.append(b[2])
    if jack_nside > 1:
        r_cov = np.cov(r_jack[0] - r_jack[1], rowvar=0, bias=1) * (n_jack - 1)
        b_cov = np.cov(b_jack[0] - b_jack[1], rowvar=0, bias=1) * (n_jack - 1)
        r_var = np.sqrt(np.diag(r_cov))
        b_var = np.sqrt(np.diag(b_cov))
    results.append([r_var, b_var])

    if jack_nside > 1:
        r_chi2 = calculate_chi_square(results[1][0], results[2][0], r_cov)
        b_chi2 = calculate_chi_square(results[1][1], results[2][1], r_cov)
        print "Goodness of fit for the red (lo) and blue (hi): ", r_chi2, b_chi2
    else:
        d_r = results[1][0] - results[2][0]
        d_b = results[1][1] - results[2][1]
        r_chi2 = d_r**2/np.sqrt(r_var[0]**2 + r_var[1]**2)
        b_chi2 = d_b**2/np.sqrt(b_var[0]**2 + b_var[1]**2)
    results.append([r_chi2, b_chi2])

    return results


def wprp_bins(gals, num_splits, box_size, jack_nside=3, rpmin=0.1, rpmax=20.0, Nrp=25):
    """
    Calculates the clustering on percentiles of sSFR rather than a simple
    red/blue split.

    Accepts:
        gals - numpy array with objects, their positions, and attributes
        red_split - value which separates two populations
        box_size - box_size of the objects in gals

    Returns:
        results - [r, [actual], [pred], [err], [chi2]]
            r - centers of r bins
            [actual] - list of xi's for each bin in sSFR
            [pred] - list of xi's for each bin in predicted sSFR
            [err] - list of errorbars for each bin in sSFR
            [chi2] - list of chi^2 values for each bin in sSFR
    """
    n_jack = jack_nside ** 2
    percentiles = [np.round(100. * i/(num_splits + 1)) for i in xrange(0, num_splits + 2)]
    bins = np.percentile(gals['ssfr'], percentiles)

    actual_dfs, pred_dfs = [], []
    for i in range(len(bins) - 1):
        actual_dfs.append(gals[(gals.ssfr > bins[i]) & (gals.ssfr < bins[i+1])])
        pred_dfs.append(gals[(gals.pred > bins[i]) & (gals.pred < bins[i+1])])

    r, rbins = make_r_scale(rpmin, rpmax, Nrp)
    results = [r]
    jacks = []
    chi2s = []
    jack_covs = []
    for dfs in [actual_dfs, pred_dfs]:
        temp = []
        temp_jack = []
        for df in dfs:
            wp = calculate_xi(df, box_size, True, jack_nside, rpmin, rpmax, Nrp)
            temp.append(wp[0])
            if jack_nside <= 1:
                temp_jack.append(wp[1])
            else:
                temp_jack.append(wp[2])
        jacks.append(temp_jack)
        results.append(temp)
    if jack_nside <= 1:
        errs = [jack for jack in jacks]
    else:
        errs = []
        for i in xrange(num_splits + 1):
            jack_covs.append(np.cov(jacks[0][i] - jacks[1][i], rowvar=0, bias=1) * (n_jack - 1))
            errs.append(np.sqrt(np.diag(jack_covs[i])))
            chi2s.append(calculate_chi_square(results[1][i], results[2][i], jack_covs[i]))
    results.append(errs)
    results.append(chi2s)
    print chi2s

    return results  # r, ssfr, pred, errs, chi2s


def assign_mark_in_bins(y, x, bins, sorter=None):
    """
    mark on y, in bins of x
    """
    assert len(y) == len(x)
    if sorter is None:
        sorter = x.argsort()
    k = np.searchsorted(x, bins, sorter=sorter)
    print k
    assert k[0] == 0 and k[-1] == len(x)
    mark = np.empty(len(x), float)
    for i, j in zip(k[:-1], k[1:]):
        assert j > i
        sel = sorter[i:j]
        mark[sel] = rankdata(y[sel])/float(j-i)
    return mark


def calc_mcf(mark, pos, rbins, box_size):
    """
    mark : 1d ndarray, length N
    pos : 2d ndarray, shape (N, 3)
    rbins : 1d ndarray
    box_size : float
    """
    pairs = []
    rmax = rbins[-1]
    with fast3tree(pos) as tree:
        tree.set_boundaries(0, box_size)
        for i, c in enumerate(pos):
            j = tree.query_radius(c, rmax, True)
            j = j[j>i]
            pairs.extend((i, _j) for _j in j)
    pairs = np.array(pairs)

    d = pos[pairs[:,0]] - pos[pairs[:,1]]
    d[d >  (0.5*box_size)] -= box_size
    d[d < (-0.5*box_size)] += box_size
    d *= d
    d = np.sqrt(d.sum(axis=-1))
    s = d.argsort()
    k = np.searchsorted(d, rbins, sorter=s)
    del d

    # make mark_rank to span -1 to +1
    mark_rank = rankdata(mark)
    mark_rank -= 1.0
    mark_rank *= (2.0/float(len(mark)-1))
    mark_rank -= 1.0

    mcf = []
    for i, j in zip(k[:-1], k[1:]):
        if j==i:
            mcf.append(np.nan)
        else:
            ii, jj = pairs[s[i:j]].T
            mcf.append((mark_rank[ii]*mark_rank[jj]).mean())

    return np.array(mcf)


def jackknife_mcf(gals, x='ssfr', y='mstar', box_size=250.0,
                  mbins=np.linspace(9.8, 12.6, 11)):
    """
    Wrapper around the marked correlation function code to handle jackknifing
    over different octants.
    """
    r, rbins = make_r_scale(rmin=.1, rmax=10, Nrp=10)
    octants = util.jackknife_octant_samples(gals, box_size)
    actual_mcfs, pred_mcfs = [], []
    for octant in octants:
        mark_x = assign_mark_in_bins(octant[x], octant[y], mbins)
        mark_pred = assign_mark_in_bins(octant['pred'], octant[y], mbins)
        mcf = calc_mcf(mark_x, octant[list('xyz')].view((float, 3)), rbins,
                       box_size)
        mcf_pred = calc_mcf(mark_pred, octant[list('xyz')].view((float, 3)),
                            rbins, box_size)
        actual_mcfs.append(mcf)
        pred_mcfs.append(mcf_pred)
    n_jack = len(octants)
    actual_mcfs, pred_mcfs = map(np.array, [actual_mcfs, pred_mcfs])
    actual = np.mean(actual_mcfs, axis=0)
    pred = np.mean(pred_mcfs, axis=0)
    error = np.sqrt(np.diag(np.cov(actual_mcfs - pred_mcfs, rowvar=0,
                                   bias=1))) * (n_jack - 1)
    return r, actual, pred, error


def shuffle_mcf(gals, x='ssfr', y='mstar', box_size=250.0,
                mbins=np.linspace(9.8, 12.6, 11)):
    """
    Calculates the marked correlation function on galaxies when the mark is
    shuffled. Sets the baseline for random when comparing the signal of MCF.
    """
    r, rbins = make_r_scale(rmin=.1, rmax=10, Nrp=10)
    mark = assign_mark_in_bins(gals[x], gals[y], mbins)
    num_trials = 50
    mcfs = []
    for i in xrange(num_trials):
        print i, num_trials
        np.random.shuffle(mark)
        mcfs.append(calc_mcf(mark, gals[list('xyz')].view((float, 3)), rbins, box_size))
    mcfs = np.array(mcfs)
    mean = np.mean(mcfs, axis=0)
    error = np.sqrt(np.diag(np.cov(mcfs, rowvar=0, bias=1)))
    return r, mean, error


def find_min_rhill(rs, masses, m_sec):
    """ Given a list of halos, calculates the minimal induced rhill value.
    Additionally returns the distance to and mass of that halo.

    If there are no larger halos, then return nan.

    Accepts:
        rs - distances to nearby halos
        masses - masses of nearby halos
        m_sec - mass of the secondary halo (or galaxy)

    Returns:
        rhill_min - minimal rhill value for all halos passed in
        r_halo - distance to the halo which produces rhill_min
        m_halo - mass of the halo which produces rhill_min
    """
    if len(rs) <= 0:
        return [np.nan, np.nan, np.nan]
    else:
        rhills = [r * (m_sec/ (3 * mass) ) ** (1./3)
                  for r, idx in zip(rs, masses)]
        rhill_min, idx = min([(r, i) for i, r in enumerate(rhills)])
        r_halo = rs[idx]
        m_halo = masses[idx]
        return rhill_min, r_halo, m_halo


def get_all_neighbors(pos, center, box_size):
    """Returns indices and positions of all neighbors within a half box_size of
    center in pos.

    Accepts:
        pos - list of positions (array-like)
        center - list of 3 coordinates
        box_size - periodicity of objects in pos
    Returns:
        idxs - indices of of neighbors in pos
        loc - positions of neighbors in pos
    """
    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        rmax = box_size/2 - 1e-6
        return tree.query_radius(center, rmax, periodic=box_size,
                                 output='both')

def calculate_clustering_score(gals, box_size, pos_tags=['x','y','zp'], rbins=[0,1,10]):
    """
    Counts the number of pairs in different radii bins
    """
    N = len(gals)
    pos = make_pos(gals, pos_tags)
    neighbors = np.zeros((N, len(rbins) - 1))
    r_max = rbins[-1]

    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        for i in xrange(N):
            if i % 10000 == 0:
                print i, N
            idxs, loc = tree.query_radius(pos[i], r_max, periodic=box_size,
                                          output='both')
            distances = get_cylinder_distance(pos[i], loc, box_size)
            for j in xrange(len(rbins) - 1):
                binned_neighbors = len(np.where((distances > rbins[j]) &
                                                (distances < rbins[j+1]))[0])
                neighbors[i][j] = binned_neighbors
    return neighbors


def calculate_density_score(gals, box_size, r_max=1):
    """
    Calculates the number of neighbors within r_max of each galaxy.
    """
    N = len(gals)
    pos = make_pos(gals)
    neighbors = np.zeros(N)

    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        for i in xrange(N):
            if i % 10000 == 0:
                print i, N
            count = tree.query_radius(pos[i], r_max, periodic=box_size,
                                        output='count')
            neighbors[i] = count
    return neighbors


def calculate_red_density_score(gals, box_size, r_max=1, red_cut=-11, color='ssfr'):
    """Returns count of red neighbors within r_max of each galaxy.

    Future versions will have a weighting score for neighbor galaxies.
    This could be a inverse distance weighting or one that incorporates color.

    Accepts:
        gals - catalog containing galaxy positions and colors
        box_size - periodicity of objects in pos
        r_max - radius around center whitn which to grab neighbors
    Returns:
        red_neighbors - array of red neighbor counts for each galaxy
    """
    N = len(gals)
    pos = make_pos(gals)
    red_neighbors = np.zeros(N)

    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        for i in xrange(N):
            if i % 10000 == 0:
                print i, N
            indices = tree.query_radius(pos[i], r_max, periodic=box_size,
                                        output='index')
            neighbor_colors = gals[color][indices]
            red_neighbors[i] = len(np.where(neighbor_colors < red_cut)[0])
    return red_neighbors


def calculate_r_hill(galaxies, hosts, box_size, projected=False):
    """
    Calculates Rhill by iterating over more massive halos

    Accepts:
        galaxies - numpy array with objects, their positions, and attributes
        hosts - numpy array with objects, their positions, and attributes
    """
    half_box_size = box_size/2
    N = len(galaxies)
    r_hills = np.zeros(N)
    halo_dists = np.zeros(N)
    halo_masses = np.zeros(N)

    if projected:
        pos_tags = ['x', 'y', 'zr']
    else:
        pos_tags = ['x', 'y', 'z']

    # Iterate over the halos and compare
    for i in xrange(N):
        if i % 5000 == 0:
            print i
        m_sec = galaxy['mvir'][i]
        center = [galaxy[tag][i] for tag in pos_tags]
        larger_halos = hosts[hosts['mvir'] > m_sec]
        pos = np.zeros((len(larger_halos), 3))
        for i, tag in enumerate(pos_tags):
            pos[:, i] = larger_halos[tag][:]
        # Consider all massive neighbors
        idxs, loc = get_all_neighbors(pos, center, box_size)
        rs = get_3d_distance(center, loc, box_size)
        msk = rs > 0.0
        rs, idxs = rs[msk], idxs[msk]
        masses = larger_halos['mvir'][idxs]
        rhill, halo_dist, halo_mass = find_min_rhill(rs, masses, m_sec)
        r_hills[i] = rhill
        halo_dists[i] = halo_dist
        halo_masses[i] = halo_mass

    return r_hills, halo_dists, halo_masses


def color_counts_for_HOD(id_to_bin, objects, nbins, red_cut=-11.0, id='upid', col='ssfr'):
    """ Counts the number of red and blue galaxies in bins of host halo mass.

    Accepts:
        id_to_bin - dict from halo ids to mass bins
        objects - array-like structure with galaxy/halo ids and colors
        nbins - number of mass bins
        red_cut - sSFR value which separates red/blue galaxies
        id - column name to grab associated host halo. 'upid' for satellites
             and 'id' for centrals
        col - column name for color. Usually 'ssfr' or 'pred'
    Returns:
        {blue,red}_counts - counts of galaxies associated with halos in a given
                            mass bin
    """
    blue_counts, red_counts = np.zeros(nbins), np.zeros(nbins)
    bins = pd.Series(objects[id]).map(id_to_bin)
    colors = objects[col] < red_cut
    for bin_id, red in zip(bins, colors):
        if not np.isnan(bin_id):
            if red:
                red_counts[bin_id] += 1
            else:
                blue_counts[bin_id] += 1
    return blue_counts, red_counts


def HOD(d0, test_gals, msmin=9.8, msmax=None, log_space=None):
    """
    Calculates the Halo Occupation Distribution on d0 and test_gals in a given
    stellar mass bin.

    Accepts:
        d0 - Full galaxy catalog
        test_gals - Partial galaxy catalog for which there are predicted colors
        msmin - lower limit on stellar mass
        msmax - upper limit on stellar mass (None to specify cumulative range)
        log_space - bins for mass of host halos
    Returns:
        results - [centers, [actual], [pred]]
            centers - host halo mass bin values
            [actual] - HOD for centrals (red/blue) and satellites (red/blue)
            [pred] - predicted HOD for centrals (red/blue) and satellites (red/blue)

    """
    mvir = d0['mvir']
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
    for i in xrange(len(halos)):
        bin_id = np.digitize([halos['mvir'][i]], edges, right=True)[0]
        halo_id_to_bin[halos['id'][i]] = min(bin_id, nbins-1)

    num_actual_blue_s, num_actual_red_s = color_counts_for_HOD(halo_id_to_bin, satellites, nbins, id='upid', col='ssfr')
    num_actual_blue_c, num_actual_red_c = color_counts_for_HOD(halo_id_to_bin, centrals, nbins, id='id', col='ssfr')

    pred_sats = test_gals[test_gals['upid'] != -1]
    pred_cents = test_gals[test_gals['upid'] == -1]
    num_pred_blue_s, num_pred_red_s = color_counts_for_HOD(halo_id_to_bin, pred_sats, nbins, id='upid', col='pred')
    num_pred_blue_c, num_pred_red_c = color_counts_for_HOD(halo_id_to_bin, pred_cents, nbins, id='id', col='pred')

    results = []
    results.append(centers)
    results.append([[num_actual_red_c/num_halos, num_actual_blue_c/num_halos], [num_actual_red_s/num_halos, num_actual_blue_s/num_halos]])
    results.append([[num_pred_red_c/num_halos, num_pred_blue_c/num_halos], [num_pred_red_s/num_halos, num_pred_blue_s/num_halos]])
    return results


def HOD_wrapper(df, test_gals, box_size):
    """
    Splits up the galaxies into octants and passes the data to HOD
    """
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


def radial_conformity(centrals, neighbors, msmin, msmax, box_size, rbins,
        satellites=False, red_cut=-11, col='ssfr'):
    """
    Calculates quenched fraction of satellites of quenched/star-forming
    centrals binned by radius.
    """
    rmin, rmax = np.min(rbins), np.max(rbins)
    nrbins = len(rbins) - 1
    all_central_nbr_counts = [[] for _ in xrange(nrbins)]
    q_central_nbr_counts = [[] for _ in xrange(nrbins)]
    sf_central_nbr_counts = [[] for _ in xrange(nrbins)]

    n_pos = make_pos(neighbors)
    with fast3tree(n_pos) as tree:
        for c_pos, c_color, c_id in zip(centrals[list('xyz')], centrals[col],
                                        centrals['id']):
            idx, pos = tree.query_radius(list(c_pos), rmax, periodic=box_size, output='both')
            distances = get_3d_distance(c_pos, pos, box_size)
            for ii, dist in zip(idx, distances):
                if satellites:
                    if neighbors['upid'] is not c_id:
                        continue
                if dist < rmin or dist > rmax:
                    continue
                rbin = np.digitize([dist], rbins, right=True)[0] - 1
                nbr_red = centrals[ii][col] < red_cut
                if c_color < red_cut:
                    q_central_nbr_counts[rbin].append(nbr_red)
                else:
                    sf_central_nbr_counts[rbin].append(nbr_red)
                all_central_nbr_counts[rbin].append(nbr_red)

    def quenched_neighbor_fraction(nbr_counts):
        return np.array([np.mean(count) for count in nbr_counts])

    return quenched_neighbor_fraction(q_central_nbr_counts), \
        quenched_neighbor_fraction(sf_central_nbr_counts), \
        quenched_neighbor_fraction(all_central_nbr_counts)


def radial_conformity_wrapper(gals, box_size, msmin, msmax, red_cut=-11,
                              cols=['ssfr', 'pred']):
    octants = util.jackknife_octant_samples(gals, box_size)
    results = []
    predictions = []
    rmin, rmax, nrbins = 0.1, 10.0, 10
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nrbins+1)
    r = np.sqrt(rbins[1:] * rbins[:-1])
    for i, sample in enumerate(octants):
        centrals = sample[sample['upid'] == -1]
        centrals = centrals[np.where((centrals['mstar'] > msmin) &
                                     (centrals['mstar'] < msmax))[0]]

        print i
        results.append(radial_conformity(centrals, centrals, msmin, msmax, box_size, rbins,
                                         False, red_cut, cols[0]))
        predictions.append(radial_conformity(centrals, centrals, msmin, msmax, box_size,
                                             rbins, False, red_cut, cols[1]))
    red_fqs, blue_fqs, all_fqs = zip(*results)
    red_fqs_pred, blue_fqs_pred, all_fqs_pred = zip(*predictions)
    actual = np.mean(red_fqs, axis=0), np.mean(blue_fqs, axis=0), np.mean(all_fqs, axis=0)
    pred = np.mean(red_fqs_pred, axis=0), np.mean(blue_fqs_pred, axis=0), np.mean(all_fqs_pred, axis=0)
    actual_err = np.sqrt(np.diag(np.cov(red_fqs, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(all_fqs, rowvar=0, bias=1)))
    pred_err = np.sqrt(np.diag(np.cov(red_fqs_pred, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs_pred, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs_pred, rowvar=0, bias=1)))
    return r, actual, pred, actual_err, pred_err


def satellite_conformity(gals, msmin, msmax, red_cut=-11, col='ssfr'):
    """
    Calculates quenched fraction of satellites of quenched/star-forming centrals.
    Only gives the overall fraction (does not include radial dependence)
    """
    centrals = gals[gals['upid'] == -1]
    centrals = centrals[np.where((centrals['mstar'] > msmin) &
                                 (centrals['mstar'] < msmax))[0]]
    red_c = centrals[centrals[col] < red_cut]
    blue_c = centrals[centrals[col] > red_cut]

    def quenched_satellite_fraction(cents):
        ids = set(cents['id'])
        sats = gals[gals['upid'] in ids]
        ssfrs = sats[col]
        f_q = np.mean(ssfrs < red_cut)
        return f_q

    return quenched_satellite_fraction(red_c), \
        quenched_satellite_fraction(blue_c)


def satellite_conformity_wrapper(gals, box_size, msmin, msmax, red_cut=-11,
                                cols=['ssfr', 'pred']):
    octants = util.jackknife_octant_samples(gals, box_size)
    results = []
    predictions = []
    rmin, rmax, nrbins = 0.1, 10.0, 10
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nrbins+1)
    r = np.sqrt(rbins[1:] * rbins[:-1])
    for i, sample in enumerate(octants):
        centrals = sample[sample['upid'] == -1]
        centrals = centrals[np.where((centrals['mstar'] > msmin) &
                                     (centrals['mstar'] < msmax))[0]]
        satellites = sample[sample['upid'] != -1]

        print i
        results.append(radial_conformity(centrals, satellites, msmin, msmax, box_size, rbins,
                                         True, red_cut, cols[0]))
        predictions.append(radial_conformity(centrals, satellites, msmin, msmax, box_size,
                                             rbins, True, red_cut, cols[1]))
    red_fqs, blue_fqs, all_fqs = zip(*results)
    red_fqs_pred, blue_fqs_pred, all_fqs_pred = zip(*predictions)
    actual = np.mean(red_fqs, axis=0), np.mean(blue_fqs, axis=0), np.mean(all_fqs, axis=0)
    pred = np.mean(red_fqs_pred, axis=0), np.mean(blue_fqs_pred, axis=0), np.mean(all_fqs_pred, axis=0)
    actual_err = np.sqrt(np.diag(np.cov(red_fqs, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(all_fqs, rowvar=0, bias=1)))
    pred_err = np.sqrt(np.diag(np.cov(red_fqs_pred, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs_pred, rowvar=0, bias=1))), \
        np.sqrt(np.diag(np.cov(blue_fqs_pred, rowvar=0, bias=1)))
    return r, actual, pred, actual_err, pred_err


def radial_profile_counts(gals, hosts, box_size, r, rbins, rmax, col='ssfr'):
    """ Calculates the distribution of gals around hosts as a function of r
    """
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
        for j in xrange(len(hosts)):
            num_red, num_blue = np.zeros(len(r)), np.zeros(len(r))
            num_pred_red, num_pred_blue = np.zeros(len(r)), np.zeros(len(r))
            center = [hosts[tag][j] for tag in pos_tags]
            idxs, pos = tree.query_radius(center, rmax, periodic=box_size, output='both')
            rs = get_3d_distance(center, pos, box_size=box_size)
            msk = rs > 0
            rs = rs[msk]
            idxs = idxs[msk]
            for dist, sat_idx in zip(rs, idxs):
                rbin = np.digitize([dist], rbins) - 1 # -1 for the r vs rbin
                # indexing reflects return values from tree
                if gals['ssfr'][sat_idx] < -11:
                    num_red[rbin] += 1
                else:
                    num_blue[rbin] += 1
                if gals['pred'][sat_idx] < -11:
                    num_pred_red[rbin] += 1
                else:
                    num_pred_blue[rbin] += 1
            num_reds.append(num_red)
            num_blues.append(num_blue)
            num_pred_reds.append(num_pred_red)
            num_pred_blues.append(num_pred_blue)
            diff_reds.append(num_red - num_pred_red)
            diff_blues.append(num_blue - num_pred_blue)
    all_counts = map(np.array, [num_reds, num_blues, num_pred_reds, num_pred_blues])
    means = map(lambda x: np.mean(x, axis=0), all_counts)
    stds = map(lambda x: np.std(x, axis=0), [diff_reds, diff_blues])
    for counts in means:
        results.append(counts)
    for errs in stds:
        results.append(errs)
    return results


def radial_profile(hosts, gals, box_size, rmin=0.1, rmax=5.0, Nrp=10):
    """ Wrapper around radial_profile_counts for different mass bins of halos.
    """
    r, rbins = make_r_scale(rmin, rmax, Nrp)
    results = [r]
    gals = gals[gals.upid != -1]

    mvir = hosts['mvir']
    delta = 0.25
    masses = [12,13,14]
    mlims = [[10**(mass-delta), 10**(mass+delta)] for mass in masses]

    for lim in mlims:
        host_sel = hosts[(hosts.mvir > lim[0]) & (hosts.mvir < lim[1])]
        host_ids = set(host_sel['id'])
        gal_sel = gals.copy()
        gal_sel = util.add_column(gal_sel, 'include',
                        [upid in host_ids for upid in gal_sel['upid']])
        gal_sel = gal_sel[gal_sel['include'] == True]
        counts = radial_profile_counts(gal_sel, host_sel, box_size, r, rbins, rmax)
        results.append(counts)

    return results


def counts_for_fq(df, red_cut, col, x_vals, bins):
    """Histogram of red and blue galaxies binned by x_vals
    """
    sel = np.where(df[col])
    red_sel = np.where(df[col] < red_cut)[0]
    blue_sel = np.where(df[col] > red_cut)[0]
    red_counts,_ = np.histogram(x_vals[red_sel], bins)
    blue_counts,_ = np.histogram(x_vals[blue_sel], bins)
    return [red_counts.astype(float), blue_counts.astype(float)]


def quenched_fraction(gals, red_cut, host_mass_dict, mbins=None, nbins=14):
    """ Calculates quenched fraction as a function of host halo mass
    """
    cents = gals[gals.upid == -1]
    sats = gals[gals.upid != -1]
    central_masses = cents['mvir']
    host_ids = set(host_mass_dict.keys())
    #sats = sats[sats.upid.isin(host_ids)]
    sats = sats[np.where(sats['upid'] in host_ids)[0]]
    sat_host_masses = np.array([host_mass_dict[upid] for upid in sats['upid']])

    if mbins is None:
        masses = gals.mvir
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
    """ Splits the quenched fraction into octants to calculate jackknife errors
    """
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
    """ Calculates the quenched fraction binned by Sigma 5
    """
    sections = [gals[(gals['mstar'] >= cutoffs[i]) &
                     (gals['mstar'] < cutoffs[i+1])]
                for i in xrange(len(cutoffs)-1)]
    s5 = gals['s5']
    print len(s5)
    if dbins is None:
        dbins = np.linspace(min(s5), max(s5), nbins)
    centers = (dbins[:-1] + dbins[1:])/2 # average because we're already in logspace
    results = [cutoffs, centers]

    for col in ['ssfr', 'pred']:
        temp = []
        for section in sections:
            dists = section['s5']
            temp.append(counts_for_fq(section, red_cut, col, dists, dbins))
        results.append(temp)
    return results


def density_vs_fq_wrapper(gals, box_size, cutoffs, red_cut=-11.0, nbins=10):
    """ Calculates quenched fraction vs density in octants to include JK errors
    """
    octants = util.jackknife_octant_samples(gals, box_size)
    s5 = gals['s5']
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


def calculate_distorted_z(df):
    """Calculates redshift distortion used in redshift space distances.
    """
    zp = df['z'] + df['vz']/100
    return util.add_column(df, 'zp', zp)


def calculate_redshift(df):
    """Calculates redshifts based on a line of sight.
    """
    c = 3e5     # km/s
    table = generate_z_of_r_table(0.3, 0.7)
    zred = z_of_r(df['z'], table)
    df['zr'] = zred + df['vz']/c
    return df


def calculate_N_gal_bolshoi(df, debug=True):
    udf = df.sort_values(by='upid')
    udf = udf.reset_index(drop=True)
    counts = Counter()
    id_to_idx = defaultdict(list)
    for idx, row in udf.iterrows():
        if row.upid == -1:
            counts[row.id] += 1
            id_to_idx[row.id].append(idx)
        else:
            counts[row.upid] += 1
            id_to_idx[row.upid].append(idx)
    udf['n_gal'] = np.zeros(len(udf))
    for halo_id, n_gal in counts.items():
        h_idxs = id_to_idx[halo_id] # list of indices of children
        udf.ix[h_idxs,'n_gal'] = n_gal # n_gal should be number of children shared by host (same for all children)
        # TODO: confirm that ngal should be shared - also doesn't matter because of host select
    if debug:
        all_host_ids = set(df[df.upid == -1].id.values)
        num_no_host = 0
        for i, row in df.iterrows():
            if row.upid != -1 and row.upid not in all_host_ids:
                num_no_host += 1
        print "number of galaxies without a host:", num_no_host
    dc = df.sort_values(by='upid')
    dc['n_gal'] = udf['n_gal'].values
    dc.sort_index(inplace=True)
    df['n_gal'] = dc['n_gal'].values
    print "Done"
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

    if nz == 1:
        for i in range(1, npts):
            if (table['r'][i] > r): break
        slope = (table['z'][i] - table['z'][i-1])/(table['r'][i]-table['r'][i-1])
        zred = table['z'][i-1] + slope*(r-table['r'][i-1])
    else:
        for i in range(1, npts):
            ii, = np.where((r >= table['r'][i-1]) & (r < table['r'][i]))
            count = len(ii)
            if count == 0: continue
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
