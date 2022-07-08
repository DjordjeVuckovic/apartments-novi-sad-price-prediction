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
from sklearn.metrics import confusion_matrix

pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('categoric_data.csv')
    X_ = data.drop('Price(EUR)', axis=1)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    sns.histplot(data['Price(EUR)'], kde=True)
    plt.title('Categoric count')
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Train score: ", clf.score(X_train, y_train))
    print("Test score: ", clf.score(X_test, y_test))

    feature_imp = pd.Series(clf.feature_importances_, index=X_.columns).sort_values(ascending=False)
    print(feature_imp[:15])
    sort = clf.feature_importances_.argsort()
    fig = plt.figure(figsize=(15, 5))
    plt.barh(X_.columns[sort[-10:]], clf.feature_importances_[sort[-10:]])
    plt.xlabel("Feature Importance")
    plt.title("Features")
    plt.show()
    print("Confusion matrix")
    cm = confusion_matrix(y_test, preds, labels=[1, 2, 3])
    print(cm)
