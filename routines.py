import numpy as np
import matplotlib.pyplot as plt
from fast3tree import fast3tree
from scipy.stats import chisquare
from sklearn import preprocessing
import seaborn as sns
#sns.set_context('poster')
#sns.set(font_scale=3, style='whitegrid')
from CorrelationFunction import projected_correlation
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from itertools import chain
from collections import defaultdict
import os
import errno
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


sns.set(font_scale=3.5, rc={'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25})
#sns.set_style('whitegrid')
sns.set_style('ticks')

# Box parameters
h = 0.7
L = 250.0/h
zmax = 40.0
box_size = 250.0

# Parameters used for correlation function plots
rpmin = 0.1
rpmax = 20.0
Nrp = 25
rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
r = np.sqrt(rbins[1:]*rbins[:-1])

png = '.png'
red_col, blue_col = sns.xkcd_rgb['reddish'], sns.xkcd_rgb['blue']


########################################
# Tools for smaller physics calculations
########################################

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


def get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=1,
                             exclude_self=False):
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


## TODO: remove this
def count_neighbors_within_r(center, tree, box_size, r):
    """
    Queries the tree for all objects within r of a given center and returns the
    count of objects
    """
    half_box_size = box_size/2.0
    tree.set_boundaries(0.0, box_size)
    rfid = tree.query_nearest_distance(center)
    #if rfid == 0.0:
        #rfid = box_size/np.power(tree.data.shape[0], 1.0/3.0)*10.0
        #if rfid > half_box_size:
            #rfid = half_box_size - 2e-6
    #rfid += 1e-6
    if r > half_box_size:
        r = half_box_size
    idx, pos = tree.query_radius(center, r, periodic=box_size, output='both')
    return len(idx) - 1 #exclude self


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


def find_min_rhill(rs, idxs, m_sec, larger_halos):
    if len(rs) < 0:
        return [np.nan, np.nan, np.nan]
    else:
        rhills = [r * (m_sec/ (3 * larger_halos['mvir'][idx]) ) ** (1./3)
                  for r, idx in zip(rs, idxs)]
        rhill_min = min(rhills)
        idx = np.argmin(rhills)
        r_halo = rs[idx]
        m_halo = larger_halos['mvir'][idx]
        return rhill_min, r_halo, m_halo


def calculate_r_hill(galaxies, hosts, box_size, projected=False):
    """
    calculate_r_hill iterates over all halos in the halo set. For each halo,
    calculate the rhill generated by the 10 more massive nearest neighbors in
    massive_halos, save the r_hill min and corresponding distance and mass of
    the determining neighbor halo.
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
        with fast3tree(pos) as tree:
            tree.set_boundaries(0.0, box_size)
            rmax = half_box_size - 1e-6
            idxs, pos = tree.query_radius(center, rmax, periodic=box_size, output='both')
        dx = get_distance(center[0], pos[:, 0], box_size=box_size)
        dy = get_distance(center[1], pos[:, 1], box_size=box_size)
        dz = get_distance(center[2], pos[:, 2], box_size=box_size)
        r2 = dx*dx + dy*dy + dz*dz
        msk = r2 > 0.0
        rs = np.sqrt(r2[msk])
        idxs = idxs[msk]
        rhill, halo_dist, halo_mass = find_min_rhill(rs, idxs, m_sec, larger_halos)
        r_hills.append(rhill)
        halo_dists.append(halo_dist)
        halo_masses.append(halo_mass)

    return np.array(r_hills), np.array(halo_dists), np.array(halo_masses)


def r_hill_pdfs(rhills, dists, masses):
    data = [rhills, dists, masses]
    xlabels = ['$Log(Rhill)$','$Log(r_{Halo_{Rhill}})$','$Log(M_{Halo_{Rhill}})$']
    for dat, xlab in zip(data, xlabels):
        plt.figure()
        sns.distplot(np.log10(dat[~np.isnan(dat)]), kde=False, norm_hist=True)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.ylabel('$PDF$')
        plt.xlabel(xlab)
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # labels = ['$10^{' + str(label) + '}$' for label in np.arange(-5,4)]
        # ax.set_xticklabels(labels)


########################################
# Tools for exploratory data analysis
########################################
# code to do nearest nbr's with fast3tree
# pull code from here
#    https://bitbucket.org/beckermr/fast3tree

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


def get_dist_and_attrs(hosts, gals, nn, attrs, projected=False):
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


def scale_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data), scaler


def split_test_train(d, box_size, fraction=0.125):
    d_train = d.sample(frac=fraction)
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


def plot_differences_3d(x1, x2, predicted, actual, name):
    """
    Given two arrays of input data, create a 3D scatterplot with marker sizes
    proportional to the difference in predicted and actual values.
    """
    num_samples = len(x1)
    diffs = predicted - actual

    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x1, x2, predicted, c='r')
    ax.scatter(x1, x2, actual, c='b', alpha=0.3)
    plt.title(name + " Actual vs Predicted ({})".format(num_samples))
    plt.legend(["Predicted", "Actual"], fontsize=6, loc='lower left')
    ax.set_xlabel('distance')
    ax.set_ylabel('mvir')
    ax.set_zlabel('ssfr')

    # Scatterplot with size indicating how far it is from the true estimation

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x1, x2, predicted, c=red_col, s=5 * np.exp(2 * np.abs(diffs)), alpha=0.7)
    plt.title(name + " Predictions with errors")
    ax.set_xlabel('distance')
    ax.set_ylabel('mvir')
    ax.set_zlabel('ssfr')


def plot_kdes(predicted, actual, name):
    """
    Individual smooth histograms of ssfr distribution. Ensures that the number
    of red vs blue galaxies is reasonable.
    """
    with sns.color_palette("pastel"):
        sns.kdeplot(actual, shade=True, label='input', clip=(-14,-8.5))
        sns.kdeplot(predicted, shade=True, label='Predicted', clip=(-14,-8.5))
        title = 'KDE of ssfr ({})'.format(name)
        #plt.title(title)
        plt.xlabel('sSFR')
        #plt.savefig(image_prefix + title + png)


def cross_scatter_plot(predicted, actual, name=''):
    """
    Heatmap of how well individual predictions do. Correlation coefficient r
    is included in the plot.
    """
    lims = [-13, -9]
    print lims
    g = sns.jointplot(actual, predicted, color=sns.xkcd_rgb['jade'], xlim=lims,
            ylim=lims, kind='hex', size=10)
    g.set_axis_labels("Actual", "Predicted")
    plt.colorbar()
    #plt.savefig(image_prefix + 'Scatter ' + name + png)


def sample_model(model, name, Xtrain, ytrain, Xtest, ytest, num_samples=500):
    """
    Combines a handful of the sanity checks and plots.
    """
    model.fit(Xtrain, ytrain)
    sel = np.random.permutation(len(Xtest))[:num_samples]
    # dist, mvir = Xtest[sel, 0], Xtest[sel, 1]
    s = ytest[sel]

    # Plot the predictions and compare
    results = model.predict(Xtest[sel])
    # diffs = results - s

    # plot_differences_3d(dist, mvir, results, s, name)
    name += ' ' + str(len(sel))
    plot_kdes(results, s, name)

    cross_scatter_plot(results, s, name)
    # Scatterplot comparison between actual and predicted
    print "Explained variance score is: ", explained_variance_score(results, s)

    plt.show()


def y_tick_formatter(x, pos):
    s = '%s' % Decimal("%.1g" % x)
    return s


########################################
# Tests aka Heavy Lifters
########################################

def plot_rwp(gals, red_split, box_size, cols=['ssfr', 'pred'],
             rpmin=0.1, rpmax=20.0, Nrp=25):
    rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
    r = np.sqrt(rbins[1:]*rbins[:-1])
    a_xis, a_var, p_xis, p_var = wprp_red_blue(d_test, cols, red_split,
                                               box_size, rpmin, rpmax, Nrp)
    fig = plt.figure(figsize=(12, 12))
    plt.xscale('log')
    plt.errorbar(r, r * a_xis[0], r*a_var[0], fmt='-o', color=red_col)
    plt.errorbar(r, r * a_xis[1], r*a_var[1], fmt='-o', color=blue_col)
    plt.errorbar(r, r * p_xis[0], r*p_var[0], fmt='--o', color=red_col, alpha=0.6)
    plt.errorbar(r, r * p_xis[1], r*p_var[1], fmt='--o', color=blue_col, alpha=0.6)
    plt.ylabel('$r$ $w_p(r_p)$')
    plt.xlabel('$r$ $[Mpc$ $h^{-1}]$')
    plt.xlim(1e-1, 30)
    adjust_plot_ticks()


def plot_wprp(actual_xis, actual_cov, pred_xis, pred_cov, set_desc, num_splits):
    """
    Plots calculated values of the correlation function and error bars
    as well as a secondary plot with a power fit for each group
    """
    n_groups = len(actual_xis)
    # create a range of colors from red to blue
    colors = sns.blend_palette([red_col, blue_col], n_groups)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    for i, xi_pred, cov_pred, xi_actual, cov_actual in \
            zip(xrange(n_groups), pred_xis, pred_cov, actual_xis, actual_cov):

        print str(i) + 'th bin'
        print 'chi square is:', chisquare(xi_pred, xi_actual)
        var1 = np.sqrt(np.diag(cov_pred))
        var2 = np.sqrt(np.diag(cov_actual))
        plt.errorbar(r, xi_actual, var2, fmt='-o', label=str(i+1), color=colors[i])
        plt.errorbar(r, xi_pred, var1, fmt='--o', color=colors[i], alpha=0.6)


    y_format = ticker.FuncFormatter(y_tick_formatter)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(pad=20)
    #plt.ticklabel_format(axis='y', style='plain')
    title = 'wp(rp) for ' + set_desc
    #plt.title(title)
    plt.xlabel('$r$ $[Mpc$ $h^{-1}]$')
    plt.xlim(1e-1, 30)
    plt.ylabel('$w_p(r_p)$')
    #plt.legend()

    # Fits power laws of the form c(x^-1.5) + y0

    plt.figure()
    plt.subplot(121)
    plt.hold(True)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    fit_region, = np.where(r > 2)
    r_fit = r[fit_region]
    normalizations = []
    for i, xi_pred in zip(xrange(num_splits + 1), pred_xis):
        popt, pcov = curve_fit(fixed_power_law, r_fit, xi_pred[fit_region],
                               p0= [500,20])
        normalizations.append(popt[1])
        plt.plot(r_fit, fixed_power_law(r_fit, *popt), color=colors[i], label=str(i+1))
    plt.legend()

    plt.subplot(122)
    sns.barplot(np.arange(1,num_splits + 2), np.array(normalizations), palette=colors)

    #plt.savefig(image_prefix + title + png)
    plt.show()

    return


def wprp_red_blue(gals, red_split, box_size, cols=['ssfr','pred'],
                  rpmin=0.1, rpmax=20.0, Nrp=25): # for 2 splits
    results = []
    for col in cols:
        red = gals[gals.col < red_split]
        blue = gals[gals.col > red_split]
        rx, rc = calculate_xi(red, box_size, rpmin, rpmax, Nrp)
        bx, bc = calculate_xi(blue, box_size, rpmin, rpmax, Nrp)
        xis = [rx,bx]
        covs = [rc, bc]
        results.append(xis)
        results.append(covs)
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
        xis = []
        covs = []
        for df in dfs:
            xi, cov = calculate_xi(df, box_size, rpmin, rpmax, Nrp)
            xis.append(xi)
            covs.append(cov)
        results.append(xis)
        results.append(covs)

    return results # actual_xis, actual_cov, pred_xis, pred_cov


def plot_richness_scatter(gals, name, full_set):
    log_counts_a, scatter_a = richness_scatter(gals[gals['ssfr'] < -11.0], full_set)
    log_counts_p, scatter_p = richness_scatter(gals[gals['pred'] < -11.0], full_set)
    plt.plot(log_counts_a, scatter_a, 'o', label='input', color='k', markersize=7)
    plt.plot(log_counts_p, scatter_p, 'o', label='predicted', color=red_col, markersize=7)
    plt.xlabel('Log Number of red satellites')
    plt.xlabel('$<log N_{red sat}>$')
    plt.xlim(-.1,2.6)
    plt.ylim(0, np.max([np.nanmax(scatter_a),np.nanmax(scatter_p)]) +.1)
    plt.ylabel('Scatter in $M_{halo}$')
    plt.legend(loc='best')

    return


def richness_scatter(gals, full):

    mass_id_dict = {}
    hosts = full[full['upid'] == -1]
    for _, host in hosts.iterrows():
        mass_id_dict[host['id']] = host['mvir']

    subs = gals[gals['upid'] != -1]
    halo_children = subs.groupby('upid').count() # number of satellites which share the same host
    halo_ids = halo_children.index.values # id of the host
    halo_masses = pd.Series(halo_ids).map(mass_id_dict) # find the mass of the

    num_children = halo_children.values.T[0]
    #plt.plot(num_children)
    plt.ylim(0,10)

    log_num_children = np.log10(num_children)

    # Potential to make more generalized version
    # Main priority is to be able to distinguish between log(2)
    # and log(3)
    nbins = 26
    bins = np.linspace(-0.05,2.45,nbins)
    bin_ids = np.digitize(log_num_children, bins, right=True)
    bincounts = np.bincount(bin_ids) # to know length of data in each bin

    scatters = np.ones(nbins-1) * np.nan
    nb_in_use = len(bincounts)
    # Relatively inefficient.. but doesn't require multidimensional arrays of different lengths
    for i, count in zip(range(1,nb_in_use+1), bincounts[1:]):
        if count == 0:
            continue
        data = np.zeros(count+1)
        j = 0
        for bin_id,mass in zip(bin_ids, halo_masses):
            if bin_id == i:
                data[j] =  mass
                j += 1
        data = pd.Series(data)
        scatters[i-1] = data.std() /data.mean()

    return (bins[1:] + bins[:-1])/2, scatters


# TODO: test by using test_gals == d_gals
def plot_density_profile(d0, d_gals, test_gals, name, coords='default'):
    """
    Binning by mass and radius, this function makes use of the
    count_neighbors_within_r helper to show radial profiles of parent galaxies.
    """
    rmin, rmax, Nrp = 0.1, 5.0, 10
    rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
    r = np.sqrt(rbins[1:]*rbins[:-1])

    dp = d0[d0['upid'] == -1]
    # Set up a tree with the galaxy set
    dp = dp.reset_index(drop=True)
    d_gals = d_gals.reset_index(drop=True)
    pos = np.zeros((len(d_gals), 3))
    if coords == 'redshift':
        pos_tags = ['x', 'y', 'zr']
    else:
        pos_tags = ['x', 'y', 'z']
    for i, tag in enumerate(pos_tags):
        pos[:, i] = d_gals[tag][:]

    mvir = dp['mvir'].values
    mmin, mmax = np.min(mvir), np.max(mvir)
    nmbins = 3
    mbins = np.logspace(np.log10(mmin), np.log10(mmax), nmbins+1)
    num_halos, _ = np.histogram(mvir, mbins)
    dp['mbin_idx'] = np.digitize(mvir, mbins)
    print mbins

    with fast3tree(pos) as tree:
        tree.set_boundaries(0.0, box_size)
        for i in xrange(nmbins):
            mass_select = dp[dp['mbin_idx'] == i]
            num_blue_actual = np.zeros(len(rbins))
            num_blue_pred = np.zeros(len(rbins))
            num_red_actual = np.zeros(len(rbins))
            num_red_pred = np.zeros(len(rbins))

            # change the order of the loop after querying the radius
            for j, halo in mass_select.iterrows():
                center = [halo[tag] for tag in pos_tags]
                idxs, pos = tree.query_radius(center, rmax, periodic=box_size, output='both')
                dx = get_distance(center[0], pos[:, 0], box_size=box_size)
                dy = get_distance(center[1], pos[:, 1], box_size=box_size)
                dz = get_distance(center[2], pos[:, 2], box_size=box_size)
                r2 = dx*dx + dy*dy + dz*dz
                msk = r2 > 0.0
                q = np.argsort(r2[msk])
                rs = np.sqrt(r2[msk][q])
                idxs = idxs[msk][q]
                for dist, sat_idx in zip(rs, idxs):
                    rbin = np.digitize([dist], rbins)

                    if d_gals['ssfr'].values[sat_idx] < -11:
                        num_red_actual[rbin] += 1
                    else:
                        num_blue_actual[rbin] += 1
                    test_gal = test_gals[test_gals['id'] == d_gals['id'].values[sat_idx]]
                    if len(test_gal):
                        if test_gal['pred'].values[0] < -11:
                            num_red_pred[rbin] += 1
                        else:
                            num_blue_pred[rbin] += 1

            volumes = [4./3 * np.pi * r**3 for r in rbins]
            num_red_actual /= num_halos[i]
            num_blue_actual /= num_halos[i]
            num_red_pred /= num_halos[i]
            num_blue_pred /= num_halos[i]
            num_red_actual /= volumes
            num_blue_actual /= volumes
            num_red_pred /= volumes
            num_blue_pred /= volumes
            plt.figure(i)
            plt.loglog(rbins, num_red_actual, '--', color=red_col, lw=4, label='input')
            plt.loglog(rbins, num_red_pred, '--', color=red_col, label='pred', alpha=0.5)
            plt.loglog(rbins, num_blue_actual, '--', color=blue_col, lw=4, label='input')
            plt.loglog(rbins, num_blue_pred, '--', color=blue_col, label='pred', alpha=0.5)
            #plt.legend(loc='best')
            plt.xlabel('$r$ $[Mpc$ $h^{-1}]$')
            plt.ylabel('$\rho_{halo}$')
            #plt.title('Radial density {:.2f}'.format(mbins[i]) + ' < mvir < {:.2f}'.format(mbins[i+1]))

    return



def plot_HOD(d0, test_gals, name, msmin, msmax=None):
    # TODO: predictions are only on a limited bin not the entire range
    # response: it's okay to plot on a range
    # response: plot using all the predictors
    """
    Currently iterates through parent catalog and counts how many centrals are
    red/blue in a given mass bin.
    """
    # need full set of upid==-1, dp should only be used for the training
    # dp = pd.DataFrame.from_records(dp.byteswap().newbyteorder())
    # if type(d) != pd.core.frame.DataFrame:
    # d = pd.DataFrame.from_records(d.byteswap().newbyteorder())
    mvir = d0['mvir'].values
    red_cut = -11.0
    # create equal spacing on log scale
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

    num_actual_blue_s, num_actual_red_s = np.zeros(nbins), np.zeros(nbins)
    satellite_bins = pd.Series(satellites['upid']).map(halo_id_to_bin)
    satellite_cols = pd.Series(satellites['ssfr']).map(lambda x: x < red_cut)
    for bin_id, red in zip(satellite_bins, satellite_cols):
        if not np.isnan(bin_id):
            if red:
                num_actual_red_s[bin_id] += 1
            else:
                num_actual_blue_s[bin_id] += 1

    num_actual_blue_c, num_actual_red_c = np.zeros(nbins), np.zeros(nbins)
    central_bins = pd.Series(centrals['id']).map(halo_id_to_bin)
    central_cols = pd.Series(centrals['ssfr']).map(lambda x: x < red_cut)
    for bin_id, red in zip(central_bins, central_cols):
        if not np.isnan(bin_id):
            if red:
                num_actual_red_c[bin_id] += 1
            else:
                num_actual_blue_c[bin_id] += 1

    pred_sats = test_gals[test_gals['upid'] != -1]
    pred_cents = test_gals[test_gals['upid'] == -1]

    num_pred_blue_s, num_pred_red_s = np.zeros(nbins), np.zeros(nbins)
    pred_sat_bins = pd.Series(pred_sats['upid']).map(halo_id_to_bin)
    pred_sat_cols = pd.Series(pred_sats['pred']).map(lambda x: x < red_cut)
    for bin_id, red in zip(pred_sat_bins, pred_sat_cols):
        if not np.isnan(bin_id):
            if red:
                num_pred_red_s[bin_id] += 1
            else:
                num_pred_blue_s[bin_id] += 1

    num_pred_blue_c, num_pred_red_c = np.zeros(nbins), np.zeros(nbins)
    pred_cent_bins = pd.Series(pred_cents['id']).map(halo_id_to_bin)
    pred_cent_cols = pd.Series(pred_cents['pred']).map(lambda x: x < red_cut)
    for bin_id, red in zip(pred_cent_bins, pred_cent_cols):
        if not np.isnan(bin_id):
            if red:
                num_pred_red_c[bin_id] += 1
            else:
                num_pred_blue_c[bin_id] += 1


    plt.figure(figsize=(14,14))
    plt.hold(True)
    plt.grid(True)
    plt.subplot(221)

    # comparison of combined HODS
    total_occupants_actual = num_actual_blue_s + num_actual_red_s + num_actual_red_c + num_actual_blue_c

    total_occupants_pred = num_pred_blue_s + num_pred_red_s + num_pred_red_c + num_pred_blue_c
    p_scale = 8./7

    #plt.title('Combined HOD (' + name + ')')
    plt.loglog(centers, (total_occupants_actual)/num_halos, color='k', lw=4, label='input')
    # Double the number of occupants because our test sample was half the catalog
    plt.loglog(centers, p_scale * (total_occupants_pred)/num_halos, color='k', label='predicted', alpha = 0.6)
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$<N_{tot}>$')
    plt.xlim(1e11, 1e15)
    plt.ylim(1e-2, 1e3)
    plt.legend(loc='best')

    plt.subplot(222)
    #plt.title('HOD red v blue (' + name + ')')
    # comparison of red and blues
    plt.loglog(centers, (num_actual_blue_s + num_actual_blue_c)/num_halos, color=blue_col, lw=4, label='input')
    plt.loglog(centers, (num_actual_red_s + num_actual_red_c)/num_halos, color=red_col, lw=4, label='input')
    plt.loglog(centers, p_scale * (num_pred_blue_s + num_pred_blue_c)/num_halos, '--', color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_s + num_pred_red_c)/num_halos, '--', color=red_col, label='predicted', alpha=0.6)
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$<N_{tot}>$')
    plt.xlim(1e10, 1e15)
    plt.ylim(1e-2, 1e3)
    plt.legend(loc='best')

    plt.subplot(223)
    #plt.title('HOD red v blue centrals (' + name + ')')
    plt.loglog(centers, (num_actual_blue_c)/num_halos, color=blue_col, lw=4, label='input')
    plt.loglog(centers, (num_actual_red_c)/num_halos, color=red_col, lw=4, label='input')
    plt.loglog(centers, p_scale * (num_pred_blue_c)/num_halos, '--', color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_c)/num_halos, '--', color=red_col, label='predicted', alpha=0.6)
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$<N_{cen}>$')
    plt.xlim(1e10, 1e15)
    plt.ylim(1e-2, 1e3)
    plt.legend(loc='best')

    plt.subplot(224)
    #plt.title('HOD red v blue satellites (' + name + ')')
    plt.loglog(centers, (num_actual_blue_s)/num_halos, color=blue_col, lw=4, label='input')
    plt.loglog(centers, (num_actual_red_s )/num_halos, color=red_col, lw=4, label='input')
    plt.loglog(centers, p_scale * (num_pred_blue_s)/num_halos, '--', color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_s)/num_halos, '--', color=red_col, label='predicted', alpha=0.6)
    plt.xlabel('$M_{halo}$')
    plt.ylabel('$<N_{sat}>$')
    plt.xlim(1e10, 1e15)
    plt.ylim(1e-2, 1e3)
    plt.legend(loc='best')
    plt.tight_layout()

    return


def plot_p_red(masses, ytest, y_hat, name):
    """
    For every bin in mass, calculate the fraction of red vs blue galaxies.
    Additionally, the distribution of color is plotted vs mvir for both the
    predicted and actual ssfr's.
    """
    nbins = 14
    bins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nbins)


    red_cut = -11.0
    actual_red, = np.where(ytest < red_cut)
    pred_red, = np.where(y_hat < red_cut)
    actual_blue, = np.where(ytest > red_cut)
    pred_blue, = np.where(y_hat > red_cut)

    actual_red_counts, _ = np.histogram(masses[actual_red], bins)
    actual_blue_counts, _ = np.histogram(masses[actual_blue], bins)
    pred_red_counts, _ = np.histogram(masses[pred_red], bins)
    pred_blue_counts, _ = np.histogram(masses[pred_blue], bins)

    p_red_test = 1.0 * actual_red_counts / (actual_red_counts + actual_blue_counts)
    p_red_predicted = 1.0 * pred_red_counts / (pred_red_counts + pred_blue_counts)
    print "Chi square is: ", chisquare(filter_nans(p_red_predicted), filter_nans(p_red_test))

    plt.hold(True)
    center = (bins[:-1] + bins[1:]) / 2
    plt.plot(center, p_red_test, lw=4, label='input', color='k', alpha=0.6)
    plt.plot(center, p_red_predicted, '--', label='predicted', color='red', alpha=0.8)
    title = 'Fraction of Quenched Galaxies {}'.format(name)
    #plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('$M_*$')
    plt.gca().set_xscale("log")
    plt.ylabel('$F_Q$')
    plt.ylim(0,1.1)
    #plt.xlim(1e10, 1e13)
    #plt.savefig(image_prefix + title + png)

    lm = pd.Series(np.log10(masses), name='$M_*$')
    plt.figure(2)
    sns.kdeplot(lm, ytest, shade=True)
    title = 'Heatmap of mstar vs ssfr (Actual) ({})'.format(name)
    plt.ylim(-13,-8)
    #plt.title(title)
    #plt.gca().set_xscale("log")

    plt.figure(3)
    sns.kdeplot(lm, y_hat, shade=True)
    #title = 'Heatmap of mstar vs ssfr (Predicted) ({})'.format(name)
    #plt.title(title)
    #plt.gca().set_xscale("log")

    plt.show()


########################################
# Miscellaneous functions
########################################

def adjust_plot_ticks():
    plt.gca().tick_params(which='minor', axis='both', length=10, width=1)
    plt.gca().tick_params(which='major', axis='both', length=13, width=2)
    plt.gca().get_yaxis().set_tick_params(which='both', direction='in')
    plt.gca().get_xaxis().set_tick_params(which='both', direction='in')


def fits_to_pandas(df):
    return pd.DataFrame.from_records(df.byteswap().newbyteorder())


def calculate_projected_z(df):
    df['zr'] = df['z'] + df['vz']/100
    return df


def fixed_power_law(x, intercept, c):
    return intercept + c * (x ** -1.5)


def power_law(x, intercept, c, power=-1):
    return intercept + c * x ** power


def filter_nans(arr):
    return np.ma.masked_array(arr, np.isnan(arr))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
