"""
House prices prediction -- Deep Learning and Artificial Neural Networks
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    """
    Main program
    """
    # Defines training related constants
    epochs: int = 100
    history: Dict[str, List[float]] = {}

    # Loads the dataset
    dataset = pd.read_csv('kc_house_data.csv')

    # Creates the directories
    os.makedirs('recording', exist_ok=True)
    os.makedirs('recording/price', exist_ok=True)
    os.makedirs('recording/mean_absolute_error', exist_ok=True)
    os.makedirs('recording/mean_squared_error', exist_ok=True)

    # Gets separately the features and the targets
    X = dataset.iloc[:, 3:].values
    X = X[:, np.r_[0:13, 14:18]]
    y = dataset.iloc[:, 2].values

    # Splits the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scales the features
    xscaler = MinMaxScaler(feature_range=(0, 1))
    X_train = xscaler.fit_transform(X_train)
    X_test = xscaler.transform(X_test)

    # Scales the targets
    yscaler = MinMaxScaler(feature_range=(0, 1))
    y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
    y_test = yscaler.transform(y_test.reshape(-1, 1))

    # Creates a `Sequential` model instance
    model = Sequential()

    # Adds a layer instance on top of the layer stack
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=17))
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

    # Configures the model for training
    model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mean_absolute_error'])

    # Trains the model
    for epoch in range(epochs):
        training = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test))

        # Aggregates history data
        for key, value in training.history.items():
            if key in history.keys():
                history[key] += value
            else:
                history[key] = value

        # Makes predictions on the test set while reversing the scaling
        actual_price = yscaler.inverse_transform(y_test)
        predicted_price = yscaler.inverse_transform(model.predict(X_test))

        # Creates the MSE plot
        plt.plot(history['loss'], label='MSE (training data)')
        plt.plot(history['val_loss'], label='MSE (validation data)')
        plt.title('Epoch #{}'.format(epoch + 1))
        plt.ylabel('MSE value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('recording/mean_squared_error/plot_{}.png'.format(epoch + 1))
        plt.clf()

        # Creates the MAE plot
        plt.plot(history['mean_absolute_error'], label='MAE (training data)')
        plt.plot(history['val_mean_absolute_error'], label='MAE (validation data)')
        plt.title('Epoch #{}'.format(epoch + 1))
        plt.ylabel('MAE value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('recording/mean_absolute_error/plot_{}.png'.format(epoch + 1))
        plt.clf()

        # Creates the Price plot
        plt.plot(predicted_price[-25:], label='Price (training data)')
        plt.plot(actual_price[-25:], label='Price (validation data)')
        plt.ticklabel_format(axis="y", style="plain", scilimits=(0, 0), useOffset=False)
        plt.axis([
            1, 25,
            0, max(predicted_price[-25:] + actual_price[-25:])
        ])
        plt.title('Epoch #{}'.format(epoch + 1))
        plt.ylabel('Price value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('recording/price/plot_{}.png'.format(epoch + 1))
        plt.clf()

    # Computes the error rate
    error = abs(predicted_price - actual_price) / actual_price
    print("Error rate: {} ".format(np.mean(error)))
