import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('my_last_data.csv')
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    min_max_scaler = MinMaxScaler().fit(X)
    X = min_max_scaler.fit_transform(X.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                             learning_rate=0.1, loss='squared_error')
    clf.fit(X_train, y_train)
    print("GradientBoostingRegressor score: ",clf.score(X_test, y_test))
    #RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train, y_train)
    y_pr = model.predict(X_test)
    print("RandomForestRegressor: ",r2_score(y_test, y_pr))
