import pandas as pd
import numpy as np
import matplotlib.pylab as plt
pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('../dataSet/theFinalFinalPrediction.csv.')
    features_nan = [ feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes=='O']
    features_nan_numerical = [feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes!='O']

    # for feature in features_nan:
    #     print("{}: {}% missing values".format(feature, np.round(data[feature].isnull().mean(),4)))
    def replace_cat_feature(dataset,feature_nan):
        data = dataset.copy()
        data[feature_nan] = data[feature_nan].fillna('Missing')
        return data

    data = replace_cat_feature(data,features_nan)
    print(data[features_nan].isnull().sum())

    for feature in features_nan_numerical:
        mid_value = data[feature].median()
        data[feature+'WasNan']= np.where(data[feature].isnull(),1,0)
        data[feature]=data[feature].fillna(mid_value)
    print(data[features_nan_numerical].isnull().sum())

    categorical_features = [feature for feature in data.columns if data[feature].dtype == 'O']
    print(categorical_features)

    for feature in categorical_features:
        temp = data.groupby(feature)['Price(EUR)'].count()/len(data)
        temp_df = temp[temp>0.005].index
        data[feature] = np.where(data[feature].isin(temp_df),data[feature],'Rare_var')

    # for feature in categorical_features:
    #      print('Feature: {}  -- number of cat: {}'.format(feature,len(data[feature].unique())))

    for feature in categorical_features:
        print('Feature: {}  -- number of cat: {}'.format(feature,data[feature].unique()))

    data.to_csv('first_data.csv', index=False)
    #print(data.head(15))
    # for feature in features_nan_numerical:
    #     print("{}: {}% missing values".format(feature, np.round(data[feature].isnull().mean(),4)))

    # print(data[features_nan_numerical].isnull().sum())

    # categorical_features = [ feature for feature in data.columns if data[feature].dtype=='O']
    # print(categorical_features)
