import numpy as np
import calc as c
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import WMAP9
import model
import astropy.units as u
import util

data_dir = '/nfs/slac/g/ki/ki23/des/jderose/observation/sdss/vagcgroups/data/'
prefix = 'clf_groups_M'
cuts = ['18_9.4_D360', '19_9.8_D360', '20_10.2_D360']
endings = ['.groups', '.galdata_corr']  # , '.prob']


"""
in *.groups
4 group mass
6 number of satellites
10, 11, 12, ra, dec, cz of central

in *.galdata_corr
5 cz
8 log ssfr
9 stellar mass
10 ra
11 dec


"""


def cartesian_from_ra_dec(ra, dec, z):
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                 distance=Distance(z=z, cosmology=WMAP9))
    cart = c.cartesian
    return cart.x.value, cart.y.value, cart.z.value


def load_massive_halo_positions():
    """
    Returns the positions of centrals, group mass, and n_gal from group catalog
    """
    m_fnames = [data_dir + prefix + cut + endings[0] for cut in cuts]
    dats = []
    for fname in m_fnames:
        dat = np.genfromtxt(fname)
        gmass = dat.T[3]
        Ngal = dat.T[5]
        ra, dec = dat.T[9], dat.T[10]
        zr = dat.T[11]/3e5
        carts = [cartesian_from_ra_dec(r, d, z) for r, d, z in zip(ra, dec, zr)]
        x, y, z = np.array(carts).T
        dats.append(np.column_stack([gmass, Ngal, x, y, z, zr]))

    combined = np.row_stack(dats)
    types = zip(['gmass', 'Ngal', 'x', 'y', 'z', 'zr'], 6 * ['<f8'])
    return combined.view(dtype=types).reshape(len(combined))


def load_galaxy_catalog():
    """
    Returns ra, dec, cz, log ssfr, log stellar mass
    """
    g_fnames = [data_dir + prefix + cut + endings[1] for cut in cuts]
    dats = []
    for fname in g_fnames:
        dat = np.genfromtxt(fname)
        ssfr = dat.T[7]
        mstar = np.log10(dat.T[8])
        ra, dec = dat.T[9], dat.T[10]
        zr = dat.T[4]/3e5
        carts = [cartesian_from_ra_dec(r, d, z) for r, d, z in zip(ra, dec, zr)]
        x, y, z = np.array(carts).T
        dats.append(np.column_stack([ssfr, mstar, x, y, z, zr]))

    combined = np.row_stack(dats)
    types = zip(['ssfr', 'mstar', 'x', 'y', 'z', 'zr'], 6 * ['<f8'])
    return combined.view(dtype=types).reshape(len(combined))


def calculate_Sigma_mass_N_gal(gals, groups):
    massive_selection = np.where(groups['gmass'] > 5e12)[0]
    print "Number of massive groups: ", len(massive_selection)
    groups = groups[massive_selection]
    print "calculating dists"
    res = c.get_projected_dist_and_attrs(groups, gals, 1, ['Ngal'], box_size=-1)
    print "done"
    return res

print "Loading galaxies"
gals = load_galaxy_catalog()
print "Loading groups"
groups = load_massive_halo_positions()
dist, res = calculate_Sigma_mass_N_gal(gals, groups)
ngal = res[0]

gals = util.add_column(gals, 'smass', dist)
gals = util.add_column(gals, 'ngal', ngal)
features = ['ngal', 'smass', 'mstar']
target = 'ssfr'
d_train, d_test, regressor = model.trainRegressor(gals, features,
                                                  target, scaled=False)

y = gals['ssfr']
y_hat = regressor.predict(util.select_features(features, gals, target=target,
                                               scaled=False)[0])
np.save('sdss_ssfr', y)
np.save('sdss_pred', y_hat)

