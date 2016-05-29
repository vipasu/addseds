import model
import calc as c
import util
import sys
import pandas as pd
from scipy.stats import pearsonr
from collections import defaultdict

def main(cat_name, proxy):
    cat = util.get_catalog(cat_name)
    df = cat['dat']
    print "num galaxies: ", len(df)

    data_dir = cat['dir']
    fname = data_dir + proxy + '.csv'
    print 'reading from: ', fname
    density = pd.read_csv(fname, header=None)
    print "num entries: ", len(density)

    df[proxy] = density.values
    df_train, df_test, m = model.trainRegressor(df, cat['box_size'], ['mstar', proxy])
    r_value = pearsonr(df_test['pred'], df_test['ssfr'])[0]
    print "Correlation Coefficient is: ", r_value
    #TODO: Add a lock here
    try:
        stat_dict = util.load_data('statistics.pckl', data_dir)
    except:
        stat_dict = defaultdict(dict)
    if not stat_dict['pearsonr'].has_key(proxy):
        stat_dict['pearsonr'][proxy] = []
    stat_dict['pearsonr'][proxy].append(r_value)
    util.dump_data(stat_dict, 'statistics.pckl', data_dir)


if __name__ == "__main__":
    cat = sys.argv[1]
    proxy = sys.argv[2]
    main(cat, proxy)

