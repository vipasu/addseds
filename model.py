import numpy as np
import pandas as pd
import util
from sklearn.tree import DecisionTreeRegressor


def trainRegressor(df, box_size, features, model, scaled=False):
    d_train, d_test = util.split_test_train(df, box_size)
    Xtrain, ytrain = util.select_features(features, d_train, scaled=scaled)
    Xtest, ytest = util.select_features(features, d_test, scaled=scaled)
    regressor = model()
    regressor.fit(Xtrain, ytrain)

    y_hat = regressor.predict(Xtest)
    d_test['pred'] = y_hat
    print min(y_hat), max(y_hat)
    return d_train, d_test, model
