import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('my_last_data.csv')
    X_ = data.drop('Price(EUR)', axis=1)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    # min_max_scaler = MinMaxScaler().fit(X)
    # X = min_max_scaler.fit_transform(X.astype(float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "min_samples_split": 5,
        "learning_rate": 0.1,
        "loss": "squared_error",
    }
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    print("GradientBoostingRegressor score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        clf.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()
    # Visualizing Our predictions
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    # Perfect predictions
    plt.plot(y_test, y_test, 'r')
    plt.title("GradientBoostingRegressor prediction")
    plt.show()
    print("Train score: ", clf.score(X_train, y_train))
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df.head(20))
    feature_imp = pd.Series(clf.feature_importances_, index=X_.columns).sort_values(ascending=False)
    print(feature_imp[:15])
    sort = clf.feature_importances_.argsort()
    print(sort)
    fig = plt.figure(figsize=(15, 5))
    plt.barh(X_.columns[sort[-10:]], clf.feature_importances_[sort[-10:]])
    plt.xlabel("Feature Importance")
    plt.title("Features GradientBoostingRegressor")
    plt.show()
    # RandomForestRegressor model
    print('----------------------------------------------------------------')
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("RandomForestRegressor score: ", r2_score(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    print("The mean squared error (MSE) on test set(RandomForestRegressor): {:.4f}".format(mse))
    print('MAE(RandomForestRegressor):', mean_absolute_error(y_test, y_pred))
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print("Train score: ", model.score(X_train, y_train))
    print(df.head(20))
    # Visualizing Our predictions
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    # Perfect predictions
    plt.plot(y_test, y_test, 'r')
    plt.title("RandomForestRegressor prediction")
    plt.show()
    feature_imp = pd.Series(model.feature_importances_, index=X_.columns).sort_values(ascending=False)
    print(feature_imp[:10])
    sort = model.feature_importances_.argsort()
    print(sort)
    fig = plt.figure(figsize=(15, 20))
    plt.barh(X_.columns[sort], clf.feature_importances_[sort])
    plt.xlabel("Feature Importance RandomForestRegressor")
    plt.title("Features")
    plt.show()
