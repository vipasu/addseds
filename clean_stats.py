import util

stats_to_delete = ['chi2_blue', 'chi2_red', 'chi2_red_00_0', 'chi2_red_00_1', 'chi2_red_00_2', 'chi2_red_25_0', 'chi2_red_25_1', 'chi2_red_25_2', 'chi2_red_50_0', 'chi2_red_50_1', 'chi2_red_50_2', 'chi2_red_75_0', 'chi2_red_75_1', 'chi2_red_75_2' ]

fname = 'statistics.pckl'
for name in [ 'Becker', 'EAGLE', 'Henriques', 'HW', 'Illustris', 'Lu', 'MB-II']:
        dir = 'data/' + name + '/'
        s = util.load_data(fname, dir)
        for stat in stats_to_delete:
            del s[stat]
        util.dump_data(s, fname, dir)

