import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv('my_data.csv')
    print(data.shape)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X.astype(float))
    early_stopping = EarlyStopping()  # stop after 2 iteration if we prevent overfitting
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.2, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5,random_state=101)
    model = Sequential()
    model.add(Dense(66, activation='relu', input_dim=66))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=66, validation_data=(X_val, y_val))
    model.summary()

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='lower right')
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    y_pred = model.predict(X_test)
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Score:', metrics.explained_variance_score(y_test, y_pred))
    # Visualizing Our predictions
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    # Perfect predictions
    plt.plot(y_test, y_test, 'r')
    plt.title("Final plot")
    plt.show()
    # visualizing residuals
    fig = plt.figure(figsize=(10, 5))
    residuals = (y_test - y_pred)
    sns.distplot(residuals)
    plt.show()
