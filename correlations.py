import calc as c
import util
import plotting as p
import model
import halotools

def main(cat, cat_name):
    data = cat['dat']
    r, rbins = c.make_r_scale(.1, 20, 25)
    pair_proxies = ['c%.2f' % _ for _ in r]
    names = ['rhillmass', 'dm5e12', 's5', 'd1', 'Pall']
    proxy_list = [['rhillmass'], ['d5e12', 'm5e12'], ['s5'], ['d1'], pair_proxies]
    predicted_ssfrs = []

    for proxies, name in zip(proxy_list, names):
        data = util.load_proxies(data, 'data/' + cat_name + '/', proxies, proxies)
        features = proxies + ['mstar']
        dtrain, dtest, regressor = model.trainRegressor(data, features)
        predicted_ssfrs.append(dtest['pred'])

    log_dir = util.get_logging_dir(cat_name)
    for proxies, pred, name in zip(proxy_list, predicted_ssfrs, names):
        dtest['pred'] = pred
        util.train_and_dump_rwp(data, proxies + ['mstar'], name + '.dat', '',
                                cat['box_size'], cat['red_cut'], logging=False)
        util.train_and_dump_rwp_bins(data, proxies + ['mstar'], name + '.dat',
                                     '', cat['box_size'], num_splits=1,
                                     red_cut=cat['red_cut'], logging=False)
        xcf = c.cross_correlation_function(dtest, cat['red_cut'],
                                           box_size=cat['box_size'])
        util.dump_data(xcf, name + '_xcf.dat', log_dir)
        mcf = c.jackknife_mcf(dtest, box_size=cat['box_size'])
        util.dump_data(mcf, name + '_mcf.dat', log_dir)

        mlims = [(9.9, 10.1), (10.1, 10.3), (10.5, 10.7)]  # change for illustris
        fnames = [''.join([name, '_conformity_', str(num), '.dat']) for num in [10.0, 10.2, 10.6]]

        for mlim, fname in zip(mlims, fnames):
            res = c.radial_conformity_wrapper(dtest, cat['box_size'], mlim[0], mlim[1])
            util.dump_data(res, fname, log_dir)

    # select centrals
    # for each of the prediction proxies
        # calculate 2 pt clustering
        # calculate 2 pt clustering in stellar mass bins
        # calculate xcf
        # calculate mcf
        # conformity

if __name__ == '__main__':
    for cat_name in ['HW', 'Illustris'][:1]:
        cat = util.get_catalog(cat_name)
        main(cat, cat_name)
