import numpy as np
import matplotlib.pyplot as plt
from fast3tree import fast3tree
from scipy.stats import chisquare
from sklearn import preprocessing
import seaborn as sns
sns.set_context('poster')
#sns.set(font_scale=3, style='whitegrid')
sns.set_style('whitegrid')
#plt.rc('font', family='serif', size=40)
#plt.yaxis.label.set_size(25)
#plt.xaxis.label.set_size(25)
from CorrelationFunction import projected_correlation
from sklearn.tree import DecisionTreeRegressor
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from itertools import chain
from collections import defaultdict
import os
import errno
import pandas as pd

# Box parameters
h = 0.7
L = 250.0/h
zmax = 40.0
box_size = 250.0

# Parameters used for correlation function plots
rpmin = 0.1
rpmax = 20.0
Nrp = 20
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


def calculate_xi(cat):
    """
    Given a catalog of galaxies, compute the correlation function using
    approriate helper functions from CorrelationFunction.py
    """
    rbins = np.logspace(np.log10(rpmin), np.log10(rpmax), Nrp+1)
    pos = np.zeros((len(cat), 3), order='C')
    pos[:, 0] = cat['x']/h
    pos[:, 1] = cat['y']/h
    pos[:, 2] = cat['z']/h + cat['vz']/h/100.0
    xi, cov = projected_correlation(pos, rbins, zmax, L, jackknife_nside=3)
    return xi, cov


def calculate_r_hill(galaxy_set, full_set, rmax=5):
    half_box_size = box_size/2
    full_set = full_set.reset_index(drop=True)
    galaxy_set = galaxy_set.reset_index(drop=True)

    #r_hills = np.zeros(len(galaxy_set))
    r_hills = []

    for i, galaxy in galaxy_set.iterrows():
        if i % 5000 == 0:
            print i
        m_sec = galaxy['mvir']
        center = [galaxy['x'], galaxy['y'], galaxy['z']]
        larger_halos = full_set[full_set['mvir'] > m_sec] #.values?

        pos = np.zeros((len(larger_halos), 3))
        for i, tag in enumerate(['x', 'y', 'z']):
            pos[:, i] = larger_halos[tag][:]
        num_tries = 0
        with fast3tree(pos) as tree:
            tree.set_boundaries(0.0, box_size)
            rmax = tree.query_nearest_distance(center) + 5
            if rmax > half_box_size:
                rmax = half_box_size - 1e-6
            #print rmax
            if rmax == 0.0:
                print rmax
            while True:
                if num_tries > 3:
                    break
                idxs, pos = tree.query_radius(center, rmax, periodic=box_size, output='both')
                if len(idxs) < 2:
                    rmax *= 3.0
                    num_tries += 1
                else:
                    break
        dx = get_distance(center[0], pos[:, 0], box_size=box_size)
        dy = get_distance(center[1], pos[:, 1], box_size=box_size)
        dz = get_distance(center[2], pos[:, 2], box_size=box_size)
        r2 = dx*dx + dy*dy + dz*dz
        msk = r2 > 0.0
        rs = np.sqrt(r2[msk])
        idxs = idxs[msk]
        if len(rs) > 0:
            rhill = min([r * (m_sec/(3 * full_set['mvir'][idx]))**(1./3) for r, idx in zip(rs, idxs)])
        else:
            rhill = half_box_size
        r_hills.append(rhill)

    return r_hills








########################################
# Tools for exploratory data analysis
########################################
# code to do nearest nbr's with fast3tree
# pull code from here
#    https://bitbucket.org/beckermr/fast3tree

def catalog_selection(d0, m, msmin, msmax):
    """
    Create parent set with mvir > m
    and galaxy set with mstar in the range (msmin, msmax)
    """
    # get parent halo set
    dp = d0[(d0['upid'] == -1) & (d0['mvir'] >= m)]
    dp = dp.reset_index(drop=True)

    # make stellar mass bin
    d = d0[(d0['mstar'] >= msmin) & (d0['mstar'] <= msmax)]
    d = d.reset_index(drop=True)
    return d, dp


def get_dist_and_attrs(dp, d, nn, attrs):
    """
    dp - parent set of halos
    d - galaxy set
    nn - num neighbors
    attrs - list of attributes (i.e. ['mvir','vmax']
    """
    pos = np.zeros((len(dp), 3))
    for i, tag in enumerate(['x', 'y', 'z']):
        pos[:, i] = dp[tag][:]

    dnbr = np.zeros(len(d))
    res = [np.zeros(len(d)) for attr in attrs]
    with fast3tree(pos) as tree:
        for i in xrange(len(d)):
            if i % 10000 == 0: print i, len(d)
            center = [d['x'].values[i], d['y'].values[i], d['z'].values[i]]
            r, ind = get_nearest_nbr_periodic(center, tree, box_size, num_neighbors=nn, exclude_self=True)
            dnbr[i] = np.log10(r)
            for j, attr in enumerate(attrs):
                res[j][i] = dp[attr].values[ind]
    return dnbr, res


def scale_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data), scaler

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
    fig = plt.figure()
    with sns.color_palette("pastel"):
        sns.kdeplot(actual, shade=True, label='Actual')
        sns.kdeplot(predicted, shade=True, label='Predicted')
        title = 'KDE of ssfr ({})'.format(name)
        plt.title(title)
        plt.xlabel('Scaled ssfr Value')
        plt.savefig(image_prefix + title + png)


def cross_scatter_plot(predicted, actual, name=''):
    """
    Heatmap of how well individual predictions do. Correlation coefficient r
    is included in the plot.
    """
    lims = [min(actual), max(actual)]
    print lims
    g = sns.jointplot(actual, predicted, color=sns.xkcd_rgb['jade'], xlim=lims,
            ylim=lims, kind='hex')
    g.set_axis_labels("Actual", "Predicted")
    plt.colorbar()
    plt.savefig(image_prefix + 'Scatter ' + name + png)


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

    plt.show()

########################################
# Tests aka Heavy Lifters
########################################

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

    title = 'wp(rp) for ' + set_desc
    #plt.title(title)
    plt.xlabel('r')
    plt.ylabel('w_p(r_p)')
    plt.legend()

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

    plt.savefig(image_prefix + title + png)
    plt.show()

    return


def wprp_comparison(gals, set_desc, num_splits, mix=False):
    """
    Takes in a data frame of galaxies and bins galaxies by ssfr. The CF is
    calculated for both predicted and actual ssfr values and passed to a helper
    function for plotting.
    """
    plt.figure()
    percentiles = [np.round(100. * i/(num_splits + 1)) for i in xrange(1, num_splits + 1)]
    splits = [np.percentile(gals['ssfr'].values, p) for p in percentiles]

    # Create empty lists for 2pt functions of multiple groups
    pred_xis, pred_cov = [], []
    actual_xis, actual_cov = [], []

    # When passed a subset of galaxies, compute the CF and cov and append it
    # to the lists passed as arguments
    def append_xi_cov(xi_list, cov_list, subset):
        xi, cov = calculate_xi(subset)
        xi_list.append(xi)
        cov_list.append(cov)

    if mix:
        # keep a list of subsets to shuffle later
        pred_cats = []

    append_xi_cov(pred_xis, pred_cov, gals[gals['pred'] < splits[0]])
    append_xi_cov(actual_xis, actual_cov, gals[gals['ssfr'] < splits[0]])

    for i in xrange(1, len(splits)):
        cat_sub_pred = gals[(gals['pred'] < splits[i]) & (gals['pred'] > splits[i-1])]
        cat_sub_actual = gals[(gals['ssfr'] < splits[i]) & (gals['ssfr'] > splits[i-1])]
        if mix:
            pred_cats.append(cat_sub_pred)
        else:
            append_xi_cov(pred_xis, pred_cov, cat_sub_pred)
        append_xi_cov(actual_xis, actual_cov, cat_sub_actual)

    if mix:
        # may have to test for functionality since switching to pandas
        pred_cats.append(gals[gals['pred'] > splits[-1]])
        collection = np.concatenate(pred_cats)
        perm = np.random.permutation(len(collection))

        idx = 0
        for pred_cat in pred_cats:
            pred_cat = collection[perm[idx:idx + len(pred_cat)]]
            idx += len(pred_cat)
            append_xi_cov(pred_xis, pred_cov, pred_cat)

    append_xi_cov(pred_xis, pred_cov, gals[gals['pred'] > splits[-1]])
    append_xi_cov(actual_xis, actual_cov, gals[gals['ssfr'] > splits[-1]])

    # TODO: comment this out
    #plot_wprp(actual_xis, actual_cov, pred_xis, pred_cov, set_desc, num_splits)

    return actual_xis, actual_cov, pred_xis, pred_cov


def wprp_fraction(gals, set_desc):
    actual_xis, actual_cov, pred_xis, pred_cov = wprp_comparison(gals, set_desc, 1)
    # how to do the error, is it just the sum?
    combined_actual = actual_xis[0]/actual_xis[1]
    combined_pred = pred_xis[0]/pred_xis[1]
    # plt.plot(

    return combined_actual, combined_pred


def plot_richness_scatter(gals, name, full_set):
    log_counts_a, scatter_a = richness_scatter(gals[gals['ssfr'] < -11.0], full_set)
    log_counts_p, scatter_p = richness_scatter(gals[gals['pred'] < -11.0], full_set)
    #fig1 = plt.figure(figsize=(12,7))
    #frame1=fig1.add_axes((.1,.3,.8,.6))
    #plt.subplot(121)
    plt.plot(log_counts_a, scatter_a, 'o', label='actual', color='k', markersize=7)
    plt.plot(log_counts_p, scatter_p, 'o', label='predicted', color=red_col, markersize=7)
    #plt.title('Scatter in richness ' + name)
    plt.xlabel('Log Number of red satellites')
    plt.xlim(-.1,2.6)
    plt.ylim(0, np.max([np.nanmax(scatter_a),np.nanmax(scatter_p)]) +.1)
    plt.ylabel('Scatter in M_halo')
    plt.legend(loc='best')
    plt.grid(True)

    #plt.subplot(122)
    #frame2=fig1.add_axes((.1,.1,.8,.2))
    #series_a = pd.Series(scatter_a, index=counts_a)
    #series_p = pd.Series(scatter_p, index=counts_p)
    # scat_diff = (series_a - series_p)/series_a
    #scat_ratio = series_p/series_a
    #plt.plot(scat_diff.index, scat_diff.values, 'ob')
    #plt.plot(scat_ratio.index, scat_ratio.values, 'ob')
    #plt.title("Scatter ratios in richness for actual vs predicted")
    #plt.axhline(0)
    #plt.ylabel('Error')
    #plt.xlabel('Number of red satellites')
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


def correlation_ratio(d_test, name):
    c_actual, c_pred = wprp_fraction(d_test, name + 'mvir + dist')
    plt.semilogx(r, c_actual, label='actual', color='k')
    plt.plot(r, c_pred, label='predicted', color=blue_col)
    #plt.title('wprp quenched vs starforming ' + name)
    plt.xlabel('r')
    plt.ylabel('w_p, Q / w_p, SF')
    plt.legend()

# TODO: test by using test_gals == d_gals
def plot_density_profile(d0, d_gals, test_gals, name):
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
    for i, tag in enumerate(['x', 'y', 'z']):
        pos[:, i] = d_gals[tag][:]

    mvir = dp['mvir'].values
    mmin, mmax = np.min(mvir), np.max(mvir)
    nmbins = 3
    mbins = np.logspace(np.log10(mmin), np.log10(mmax), nmbins+1)
    num_halos, _ = np.histogram(mvir, mbins)
    dp['mbin_idx'] = np.digitize(mvir, mbins)

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
                center = halo['x'], halo['y'], halo['z']
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
                    #print dist
                    #print rbins
                    rbin = np.digitize([dist], rbins)
                    # query for large radius and then do processing in here

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
            plt.loglog(rbins, num_red_actual, color=red_col, label='actual')
            plt.loglog(rbins, num_red_pred, color=red_col, label='pred', alpha=0.5)
            plt.loglog(rbins, num_blue_actual, color=blue_col, label='actual')
            plt.loglog(rbins, num_blue_pred, color=blue_col, label='pred', alpha=0.5)
            plt.legend(loc='best')
            plt.xlabel('distance')
            plt.ylabel('<centrals + satellites>')
            plt.title('Radial density {:.2f}'.format(mbins[i]) + ' < mvir < {:.2f}'.format(mbins[i+1]))

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
    log_space = np.arange(np.log10(np.min(mvir)), np.log10(np.max(mvir)),.1)

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
    plt.loglog(centers, (total_occupants_actual)/num_halos, color='k', label='actual')
    # Double the number of occupants because our test sample was half the catalog
    plt.loglog(centers, p_scale * (total_occupants_pred)/num_halos, color='k', label='pred', alpha = 0.6)
    plt.xlabel('M_halo')
    plt.ylabel('<N_tot>')
    plt.legend(loc='best')

    plt.subplot(222)
    #plt.title('HOD red v blue (' + name + ')')
    # comparison of red and blues
    plt.loglog(centers, (num_actual_blue_s + num_actual_blue_c)/num_halos, color=blue_col, label='actual')
    plt.loglog(centers, (num_actual_red_s + num_actual_red_c)/num_halos, color=red_col, label='actual')
    plt.loglog(centers, p_scale * (num_pred_blue_s + num_pred_blue_c)/num_halos, color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_s + num_pred_red_c)/num_halos, color=red_col, label='predicted', alpha=0.6)
    plt.xlabel('M_halo')
    plt.ylabel('<N_tot>')
    plt.legend(loc='best')

    plt.subplot(223)
    #plt.title('HOD red v blue centrals (' + name + ')')
    plt.loglog(centers, (num_actual_blue_c)/num_halos, color=blue_col, label='actual')
    plt.loglog(centers, (num_actual_red_c)/num_halos, color=red_col, label='actual')
    plt.loglog(centers, p_scale * (num_pred_blue_c)/num_halos, color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_c)/num_halos, color=red_col, label='predicted', alpha=0.6)
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.xlabel('M_halo')
    plt.ylabel('<N_cen>')
    plt.legend(loc='best')

    plt.subplot(224)
    #plt.title('HOD red v blue satellites (' + name + ')')
    plt.loglog(centers, (num_actual_blue_s)/num_halos, color=blue_col, label='actual')
    plt.loglog(centers, (num_actual_red_s )/num_halos, color=red_col, label='actual')
    plt.loglog(centers, p_scale * (num_pred_blue_s)/num_halos, color=blue_col, label='predicted', alpha=0.6)
    plt.loglog(centers, p_scale * (num_pred_red_s)/num_halos, color=red_col, label='predicted', alpha=0.6)
    plt.xlabel('M_halo')
    plt.ylabel('<N_sat>')

    plt.legend(loc='best')
    plt.tight_layout()

    return


def plot_p_red(masses, ytest, y_hat, name):
    """
    For every bin in mass, calculate the fraction of red vs blue galaxies.
    Additionally, the distribution of color is plotted vs mvir for both the
    predicted and actual ssfr's.
    """
    nbins = 50
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
    plt.plot(center, p_red_test, label='actual', color='k', alpha=0.6)
    plt.plot(center, p_red_predicted, label='predicted', color='red', alpha=0.8)
    title = 'Fraction of Quenched Galaxies {}'.format(name)
    #plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('M_*')
    plt.gca().set_xscale("log")
    plt.ylabel('F_Q')
    plt.ylim(0,1.1)
    plt.savefig(image_prefix + title + png)

    lm = np.log10(masses)
    plt.figure(2)
    sns.kdeplot(lm, ytest, shade=True)
    title = 'Heatmap of mstar vs ssfr (Actual) ({})'.format(name)
    plt.title(title)
    #plt.gca().set_xscale("log")

    plt.figure(3)
    sns.kdeplot(lm, y_hat, shade=True)
    title = 'Heatmap of mstar vs ssfr (Predicted) ({})'.format(name)
    plt.title(title)
    #plt.gca().set_xscale("log")

    plt.show()


########################################
# Miscellaneous functions
########################################

def fits_to_pandas(df):
    return pd.DataFrame.from_records(df.byteswap().newbyteorder())


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
