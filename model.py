import numpy as np
import pandas as pd
import util
from sklearn.tree import DecisionTreeRegressor


def trainRegressor(df, features, target='ssfr', pred_label='pred', model=DecisionTreeRegressor, scaled=False):
    d_train, d_test = util.split_test_train(df)
    Xtrain, ytrain = util.select_features(features, d_train, target=target, scaled=scaled)
    Xtest, ytest = util.select_features(features, d_test, target=target, scaled=scaled)
    regressor = model()
    regressor.fit(Xtrain, ytrain)

    y_hat = regressor.predict(Xtest)
    d_test = util.add_column(d_test, pred_label, y_hat)
    print min(y_hat), max(y_hat)
    return d_train, d_test, regressor
