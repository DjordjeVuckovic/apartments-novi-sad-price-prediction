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
    data['Price(EUR)'] = np.log1p(data['Price(EUR)'])
    sns.histplot(data['Price(EUR)'], kde=True)
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
    # sns.pairplot(data[['Rooms'], ['Area(m2)'], ['YearOfBuild'], ['Adice'], ['Stari Grad']])
    # plt.show()
    data.dtypes.value_counts().plot.pie()
    plt.show()
    print('UNIQUE VALUES\n')
    for col in data.columns:
        print(f'{col}: {len(data[col].unique())}\n')
    print(data[data.select_dtypes(exclude='object').columns].describe())
    plt.figure(figsize=(30, 8))
    sns.heatmap(data.isnull(), cmap='flare')
    plt.title('Null values')
    plt.show()
    # Columns containing most null values
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print('Missing data:')
    print(missing_data.head(10))
