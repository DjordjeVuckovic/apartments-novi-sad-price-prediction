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
    # print(data.head())
    # data.drop(['Index', 'Stores'], axis=1, inplace=True)
    # data.info()
    # print(data.columns)
    # print(data.head())

    #data.info()
    #print(data.columns)
    #sns.pairplot(data)
    # plt.show()
    # sns.heatmap(data.corr(), annot=True)
    # plt.show()
    # max_x = data.loc[data['Area(m2)'].idxmax()]
    # print(max_x)
    # y = data['Price(EUR)']
    # # print(y)
    # data.drop(['Price(EUR)'], axis=1, inplace=True)
    # X = data[list(data.columns)]
    # print(X.columns)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    min_max_scaler = MinMaxScaler().fit(X)
    X = min_max_scaler.fit_transform(X.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    lm = Ridge(alpha=20)
    lm.fit(X_train, y_train)
    print(lm.score(X_test, y_test))
    print("Slope:")
    print(lm.coef_)
    print("Intercept:")
    print(lm.intercept_)
    intercept = lm.intercept_
    slope = lm.coef_
    # coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    # print(coeff_df)
    # predictions = lm.predict(X_test)
    # plt.scatter(y_test, predictions)
    # plt.title("Prediction")
    # plt.show()
    #
    # sns.distplot((y_test - predictions), bins=50)
    # plt.show()
    # plt.scatter(data['Price(EUR)'], data['Rooms'])
    # plt.title("Price vs Rooms")
    # plt.show()
    # plt.scatter(data['Price(EUR)'], data['Area(m2)'])
    # plt.title("Price vs Square Feet")
    # plt.show()
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                             learning_rate=0.1, loss='ls')
    clf.fit(X_train, y_train)
    print("GradientBoostingRegressor:")
    print(clf.score(X_test, y_test))
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train, y_train)
    y_pr = model.predict(X_test)
    print("RandomForestRegressor:")
    print(r2_score(y_test, y_pr))
    # print(X)
