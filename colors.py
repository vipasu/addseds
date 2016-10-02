import util
import model
import calc as c
import corner
# import plotting as p
import numpy as np


def compare_pred_ssfr(df, predicted_ssfrs, names):
    actual = df['ssfr']
    ssfrs = [actual] + predicted_ssfrs
    N = len(ssfrs)
    ssfrs = np.array(ssfrs)
    corner.corner(ssfrs.T, labels=['sSFR'] + names, range=N*[[-13.5, -9.5]])
    return


def main(data, cat_name):
    rbins, r = c.make_r_scale(.1, 20, 25)
    pair_proxies = ['c%.2f' % _ for _ in r]
    names = ['rhillmass', 'dm5e12', 's5', 'P all']
    proxy_list = [['rhillmass'], ['d5e12', 'm5e12'], ['s5'], pair_proxies]
    predicted_ssfrs = []

    for proxies, name in zip(proxy_list, names):
        data = util.load_proxies(data, 'data/' + cat_name + '/', proxies, proxies)
        features = proxies + ['mstar']
        dtest, dtrain, regressor = model.trainRegressor(data, features)
        predicted_ssfrs.append(dtrain['pred'])

if __name__ == '__main__':
    for cat_name in ['HW', 'Illustris'][:1]:
        dat = util.get_catalog(cat_name)['dat']
        main(dat, cat_name)
