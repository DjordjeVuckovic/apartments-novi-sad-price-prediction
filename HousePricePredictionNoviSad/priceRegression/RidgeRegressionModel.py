import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np

pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('my_last_data.csv')
    X_col = data.drop('Price(EUR)', axis=1)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    s_scaler = StandardScaler()
    #X = s_scaler.fit_transform(X.astype(float))
    print("Ridge regression\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    X_train = s_scaler.fit_transform(X_train.astype(float))
    X_test = s_scaler.transform(X_test.astype(float))
    regressor = Ridge(alpha=10)
    regressor.fit(X_train, y_train)
    print(regressor.score(X_test, y_test))
    print("Slope:")
    print(regressor.coef_)
    print("Intercept:")
    print(regressor.intercept_)
    intercept = regressor.intercept_
    # slope = lm.coef_
    #coeff_df = pd.DataFrame(regressor.coef_, data.drop('Price(EUR)', axis=1).columns, columns=['Coefficient'])
    #print(coeff_df)
    y_pred = regressor.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.title("Prediction")
    plt.show()
    sns.regplot(y_test, y_pred)
    plt.title("Prediction plot")
    plt.show()
    # visualizing residuals
    sns.distplot((y_test - y_pred), bins=50)
    plt.title("Residuals")
    plt.show()
    # compare actual output values with predicted values
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df.head(20))
    # evaluate the performance of the algorithm (MAE - MSE - RMSE)
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('VarScore:', metrics.explained_variance_score(y_test, y_pred))
    bx = plt.subplots(figsize=(14, 20))
    bx = sns.barplot(x=0, y=1, data=pd.DataFrame(zip(data.drop('Price(EUR)', axis=1).columns, regressor.coef_)))
    plt.xticks(rotation='vertical')
    plt.xlabel("Model Coefficient Types")
    plt.ylabel("Model Coefficient Values")
    plt.show()
    print("----------------------------------------")
    print("Lasso regression")
    lasso = Lasso(alpha=0.01)
    regressor.fit(X_train, y_train)
    print("Lasso score: ",regressor.score(X_test, y_test))
    # price and area model
    X_area = data[['Area(m2)']]
    Y_price = data['Price(EUR)']
    rg = Ridge(alpha=10)
    rg.fit(X_area, Y_price)
    print("----------------------------------------")
    print("Area only score:",rg.score(X_area, Y_price))
    coeff_df = pd.DataFrame(rg.coef_, X_area.columns, columns=['Coefficient'])
    print(coeff_df)
