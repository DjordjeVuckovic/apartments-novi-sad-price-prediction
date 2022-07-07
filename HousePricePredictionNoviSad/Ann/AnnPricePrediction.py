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
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv('my_data.csv')
    print(data.shape)
    X = data.drop('Price(EUR)', axis=1).values
    y = data['Price(EUR)'].values
    # print(y)
    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.2, random_state=101)
    s_scaler = StandardScaler()
    # s_scaler = MinMaxScaler().fit(X)
    X_train = s_scaler.fit_transform(X_train.astype(np.float))
    X_test_ = s_scaler.transform(X_test_.astype(np.float))
    X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5)
    model = Sequential()
    model.add(Dense(66, activation='relu',input_dim=66))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(66, activation='relu'))
    model.add(Dense(1,activation='relu'))
    # model.add(Dense(activation="relu", input_dim=66, units=50, kernel_initializer="he_uniform"))
    # model.add(Dense(activation="relu", units=25, kernel_initializer="he_uniform"))
    # model.add(Dense(activation="relu", units=50, kernel_initializer="he_uniform"))
    # model.add(Dense(units=1, kernel_initializer="he_uniform"))
    # model.add(Dense(units=100, input_dim=66, activation='relu'))
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dense(units=1,activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=66,validation_data=(X_val, y_val))
    model.summary()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

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
    print('VarScore:', metrics.explained_variance_score(y_test, y_pred))
    # Visualizing Our predictions
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    # Perfect predictions
    plt.plot(y_test, y_test, 'r')
    plt.title("Final plot")
    plt.show()
