import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.pandas.set_option('display.max_columns', None)
if __name__ == '__main__':
    data = pd.read_csv('my_last_data.csv')
    # print(data['Price(EUR)'].describe())
    # print(data['Rooms'].describe())
    # print(data['Floor'].unique())
    sns.histplot(data['Price(EUR)'], kde=True)
    plt.show()
    sns.histplot(np.log1p(data['Price(EUR)']), kde=True)
    plt.show()
    corr = data.corr()
    highly_corr_features = corr.index[abs(corr['Price(EUR)']) > 0.1]
    plt.figure(figsize=(10, 10))
    sns.heatmap(data[highly_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()
    print(corr['Price(EUR)'].sort_values(ascending=False).head(10))
    fig = plt.figure(figsize=(12, 10))
    # Rooms
    plt.subplot(321)
    sns.scatterplot(data=data, x='Rooms', y='Price(EUR)')
    # Area
    plt.subplot(322)
    sns.scatterplot(data=data, x='Area(m2)', y='Price(EUR)')
    # YearOfBuild
    plt.subplot(323)
    sns.scatterplot(data=data, x='YearOfBuild', y="Price(EUR)")
    plt.subplot(324)
    # Location
    sns.scatterplot(data=data, x='Adice', y="Price(EUR)")
    plt.subplot(325)
    sns.scatterplot(data=data, x='Stari Grad', y="Price(EUR)")
    plt.subplot(326)
    sns.scatterplot(data=data, x='Detelinara', y="Price(EUR)")
    plt.show()
    # corelatted params
    plt.scatter(data['Rooms'], data['Price(EUR)'])
    plt.title("Price vs Rooms")
    plt.show()
    plt.scatter(data['Area(m2)'], data['Price(EUR)'])
    plt.title("Price vs Square Feet")
    plt.show()
    plt.scatter(data['YearOfBuild'], data['Price(EUR)'])
    plt.title("Price vs YearOfBuild")
    plt.show()
    plt.scatter(data['Stari Grad'], data['Price(EUR)'])
    plt.title("Price vs Centar")
    plt.show()
    sns.boxplot(data=data['YearOfBuild'])
    plt.show()
    sns.pairplot(data[['Price(EUR)','Rooms', 'Area(m2)', 'YearOfBuild', 'Adice', 'Stari Grad','luksuzno','prizemlje','Detelinara','Spens']])
    plt.show()
    data.dtypes.value_counts().plot.pie()
    plt.show()
