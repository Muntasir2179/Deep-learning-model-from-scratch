from random import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# creating a dataset
def generate_dataset(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # dividing dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = generate_dataset(10000, 0.3)

    # building model
    model = Sequential()
    # adding input layer and first hidden layer
    model.add(Dense(units=5, input_shape=(2,), activation='sigmoid'))
    # adding output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='MSE')

    # train the model
    model.fit(x_train, y_train, epochs=100)

    # evaluate the model
    print("\nModel evaluation: ")
    model.evaluate(x_test, y_test, verbose=1)

    # make prediction
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    prediction = model.predict(data)

    print("\nSome predictions: ")
    for d, p in zip(data, prediction):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

    print()
    print(len(x_train), len(x_test), len(y_train), len(y_test))
