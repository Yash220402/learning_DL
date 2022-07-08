from random import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def generate_dataset(num_samples, test_size=0.33):
    """Generate train and test set for addition operation."""
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # create dataset
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    print(x_train, x_test, y_train, y_test)

    # model 2 -> 5 -> 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # compile model
    model.compile(optimizer=optimizer, loss="mse")

    # train model
    model.fit(x_train, y_train, epochs=100)

    # evaluate model performance
    print("\nTest Set Performance:")
    model.evaluate(x_test, y_test, verbose=2)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    prediction = model.predict(data)

    print("\nPredictions:")
    for d, p in zip(data, prediction):
        print(f"{d[0]} + {d[1]} = {p[0]}")
