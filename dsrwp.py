import sys
import numpy as np
import util


def main():
    attr = 'log_c'
    data_dir = 'data/darksky/'
    dat = np.load(data_dir + 'galaxies_1.npy')
    proxies = ['d5e12', 'm5e12']
    proxies += ['d1', 'd2', 'd5', 'd10']
    proxies += ['s1', 's2', 's5', 's10']
    print proxies
    dat = util.load_proxies(dat, data_dir + '1/', proxies, proxies)
    med = np.median(dat[attr])

    fname = ''.join(['dm,ds12510_', attr, '.dat'])
    proxyname = ''.join(['dmds12510', mass, attr])
    util.train_and_dump_rwp(dat, proxies + ['vpeak'], fname, proxyname, 'ds1',
            400, red_cut=med, logging=False, target=attr)
