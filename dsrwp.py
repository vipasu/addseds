import sys
import numpy as np
import util


def main(mass):
    attr = 'spin_bullock'
    data_dir = 'data/darksky/'
    dat = np.load(data_dir + 'galaxies_1.npy')
    proxies = ['d' + mass, 'm' + mass]
    print proxies
    dat = util.load_proxies(dat, data_dir + '1/', proxies, proxies)
    # dat = util.add_rec_column(dat, 'log_c', np.log10(dat['c']))
    # med_c = np.median(dat['log_c'])
    med = np.median(dat[attr])
    util.train_and_dump_rwp(dat, proxies + ['vpeak'], ''.join(['dm', mass, '_',
                                                               attr, '.dat']),
                            ''.join(['dm', mass, attr]), 'ds1', 400,
                            red_cut=med, logging=False, target=attr)


if __name__ == '__main__':
    main(sys.argv[1])
