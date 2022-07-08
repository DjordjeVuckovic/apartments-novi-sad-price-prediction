from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':
    data = pd.read_csv('my_data.csv')
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    scaler = preprocessing.StandardScaler().fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    model = MLPRegressor(hidden_layer_sizes=(200, 100, 100, 66,), activation='relu', max_iter=1000, solver='adam')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Score:', metrics.r2_score(y_test, y_pred))
