import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns', None)

if __name__ == '__main__':
    data = pd.read_csv('../dataSet/categoricPrediction.csv.')
    features_nan = [feature for feature in data.columns if
                    data[feature].isnull().sum() > 1 and data[feature].dtypes == 'O']
    features_nan_numerical = [feature for feature in data.columns if
                              data[feature].isnull().sum() > 1 and data[feature].dtypes != 'O']


    def replace_cat_feature(dataset, feature_nan):
        data = dataset.copy()
        data[feature_nan] = data[feature_nan].fillna('Missing')
        return data


    data = replace_cat_feature(data, features_nan)
    print(data[features_nan].isnull().sum())

    for feature in features_nan_numerical:
        mid_value = data[feature].median()
        data[feature + 'WasNan'] = np.where(data[feature].isnull(), 1, 0)
        data[feature] = data[feature].fillna(mid_value)
    print(data[features_nan_numerical].isnull().sum())
    data.drop(["Stores"], axis=1, inplace=True)
    data.drop(["Id"], axis=1, inplace=True)
    categorical_features = [feature for feature in data.columns if data[feature].dtype == 'O']
    # print(categorical_features)

    for feature in categorical_features:
        temp = data.groupby(feature)['Price(EUR)'].count() / len(data)
        temp_df = temp[temp > 0.005].index
        data[feature] = np.where(data[feature].isin(temp_df), data[feature], 'Rare_var')


    # for feature in categorical_features:
    #      print('Feature: {}  -- number of cat: {}'.format(feature,len(data[feature].unique())))

    # for feature in categorical_features:
    #     print('Feature: {}  -- number of cat: {}'.format(feature,data[feature].unique()))

    # label_encoder = LabelEncoder()
    # for feature in categorical_features:
    #     label_encoder.fit(feature)
    #     data[feature]=label_encoder.transform(feature)

    def category_onehot_multcols(multcolumns):
        df_final = data.copy()
        i = 0
        for fields in multcolumns:

            print(fields)
            df1 = pd.get_dummies(data[fields], drop_first=True)

            data.drop([fields], axis=1, inplace=True)
            if i == 0:
                df_final = df1.copy()
            else:

                df_final = pd.concat([df_final, df1], axis=1)
            i = i + 1

        df_final = pd.concat([data, df_final], axis=1)

        return df_final


    print(data['Address'].describe())
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

    data = category_onehot_multcols(categorical_features)
    #print(data.shape)

    final_df = data.loc[:, ~data.columns.duplicated()]
    print(final_df.shape)
    # final_df.info()
    print(final_df['Price(EUR)'].describe())
    min = data.loc[data['Price(EUR)'].idxmin()]
    print(min)
    final_df.to_csv('my_last_categoric_data.csv', index=False)

    # feature_scale = [feature for feature in  data.columns if feature  in ['Rooms', 'Area(m2)', 'YearOfBuild']]

    # scaler = MinMaxScaler()
    # scaler.fit(data[feature_scale])
    # MinMaxScaler(copy=True, feature_range=(0, 1))
    # data[feature_scale] = scaler.transform(data[feature_scale])

    # dataset = pd.concat([data[['Id', 'Price(EUR)']].reset_index(drop=True),
    #                   pd.DataFrame(scaler.transform(data[feature_scale]), columns=feature_scale)],
    #                  axis=1)

    # for feature in features_nan_numerical:
    #     print("{}: {}% missing values".format(feature, np.round(data[feature].isnull().mean(),4)))

    # print(data[features_nan_numerical].isnull().sum())

    # categorical_features = [ feature for feature in data.columns if data[feature].dtype=='O']
    # print(categorical_features)
