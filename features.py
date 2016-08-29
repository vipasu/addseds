from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
import util
import calc as c
from sklearn.linear_model import Lasso
import plotting as p


cat = util.get_catalog("HW")
dat = cat['dat']

proxies = ['rhillmass', 'd1', 'd2', 'd5', 'd10', 's1', 's2', 's5', 's10',
           'd5e12', 'm5e12']
# TODO: make a log flag so that m5e12 gets log scaled as well
dat = util.load_proxies(dat, 'data/HW/', proxies, proxies)

features = proxies + ['mstar']

d_train, d_test = util.split_test_train(dat)
Xtr, ytr, xtrsc, ytrsc = util.select_features(features, d_train, scaled=True)
Xts, yts, xtssc, ytssc = util.select_features(features, d_test, scaled=True)
#poly = preprocessing.PolynomialFeatures(degree=2)
#Xtr_new = poly.fit_transform(Xtr)
#Xts_new = poly.fit_transform(Xts)


gp = GaussianProcess()
gp.fit(Xtr, ytr)
y_hat = gp.predict(Xts)
#clf = Lasso(alpha=0.2)
#clf.fit(Xtr_new, ytr)
#y_hat = clf.predict(Xts_new)

y_pred = ytssc.inverse_transform(y_hat)
y_test = ytssc.inverse_transform(yts)

p.sns.kdeplot(y_pred)
p.sns.kdeplot(y_test)

d_test = util.add_column(d_test, 'pred', y_pred)

results = c.wprp_split(d_test, red_split=-11, box_size=cat['box_size'])
util.dump_data(results, 'gp.dat', util.get_logging_dir('HW'))
