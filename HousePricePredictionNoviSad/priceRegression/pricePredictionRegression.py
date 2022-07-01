import pandas as pd
import numpy as np
import matplotlib.pylab as plt
pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_json('../resources/finalApartments2.json')
    features_nan = [ feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes=='O']

    # for feature in features_nan:
    #     print("{}: {}% missing values".format(feature, np.round(data[feature].isnull().mean(),4)))




    def replace_cat_feature(dataset,feature_nan):
        data = dataset.copy()
        data[feature_nan] = data[feature_nan].fillna('Missing')
        return data

    data = replace_cat_feature(data,features_nan)
    #print(data[features_nan].isnull().sum())

    features_nan_numerical = [feature for feature in data.columns if data[feature].isnull().sum() > 1 and data[feature].dtypes != 'O']
    for feature in features_nan_numerical:
        print("{}: {}% missing values".format(feature, np.round(data[feature].isnull().mean(),4)))

    print(data)