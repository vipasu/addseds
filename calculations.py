import calc as c
import util
import sys
import pandas as pd

def main(cat_name, proxy):
    print "Loading data..."
    cats = util.load_data('cut_cats.dat', './data/')
    print "Done loading data..."
    assert cat_name in cats.keys()
    cat = cats[cat_name]
    print "Discarding extra data..."
    del cats
    dat = cat['dat']
    if proxy[0] == 's':
        dist_func = c.get_projected_dist_and_attrs
    else:
        dist_func = c.get_dist_and_attrs
    nn = int(proxy[1:])
    print "Calculating the distance to the ", nn, "th nearest neighbor"
    dist, _ = dist_func(dat, dat, nn, [], box_size=cat['box_size'])
    fname = cat['dir'] + proxy + '.csv'
    pd.Series(dist).to_csv(fname)
    print 'wrote to: ', fname


if __name__ == "__main__":
    cat = sys.argv[1]
    proxy = sys.argv[2]
    main(cat, proxy)

