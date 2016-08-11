from collections import defaultdict
import calc, util, model
import plotting as p
p.set_plotting_context()


names = ['HW', 'Becker', 'Lu', 'Henriques', 'EAGLE', 'Illustris', 'MB-II']
catalogs = [util.get_catalog(name) for name in names]
data_dirs = ['data/' + name + '/' for name in names]
dfs = [cat['dat'] for cat in catalogs]
dfs = [util.load_proxies(df, ddir, ['rhillmass'], ['rhillmass']) for df, ddir
        in zip(dfs, data_dirs)]

mlims = [(9.9, 10.1), (10.1, 10.3), (10.5, 10.7)]

msbin_catalog_dict = defaultdict(dict)
for mlim, desc in zip(mlims, ['lo', 'mid', 'hi']):
    for name, df in zip(names, dfs):
        msmin, msmax = mlim # density match(df, mlim)
        sel = np.where((df['mstar'] > msmin) & (df['mstar'] < msmax))[0]
        msbin_catalog_dict[name][desc] = df[sel]

nrow = len(names)
ncol = 3
i = 1
for name in names:
    for mbin, df in msbin_catalog_dict[name].items():
        plt.subplot(nrow, ncol, i)
        rhillmass_values = df['rhillmass']
        med = np.median(rhillmass_values)
        dense = np.where(df['rhillmass'] < med)[0]
        under_dense = np.where(df['rhillmass'] > med)[0]
        ssfr_d = df[dense]['ssfr']
        ssfr_u = df[under_dense]['ssfr']
        sns.kdeplot(ssfr_d, color=p.red_col)
        sns.kdeplot(ssfr_u, color=p.blue_col)
        i += 1


