import util
import calc
import model
import matplotlib.pyplot as plt


data_dir = 'data/Henriques'
cat = util.get_catalog("Henriques")
dat = cat['dat']
dat = util.load_proxies(dat, data_dir, ['s5'], ['s5'])
df_train, df_test, _ = model.trainRegressor(dat, cat['box_size'], ['s5'])

plt.hexbin(df_test['ssfr'], df_test['pred'])



