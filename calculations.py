import calc as c
import util
import sys
import pandas as pd

def main(cat_name, proxy):
    cat = util.get_catalog(cat_name)
    dat = cat['dat']

    if proxy[0] == 's':
        dist_func = c.get_projected_dist_and_attrs
    else:
        dist_func = c.get_dist_and_attrs
    nn = int(proxy[1:])
    print "Calculating the distance to the ", nn, "th nearest neighbor"
    dist, _ = dist_func(dat, dat, nn, [], box_size=cat['box_size'])
    fname = cat['dir'] + proxy + '.csv'
    pd.Series(dist).to_csv(fname, index=False)
    print 'wrote to: ', fname


if __name__ == "__main__":
    cat = sys.argv[1]
    proxy = sys.argv[2]
    main(cat, proxy)

