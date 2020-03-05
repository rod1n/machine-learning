from nn import Model
from nn.layers import Dense
from nn.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np

np.random.seed(1)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))
X_train = X_train / 255
y_train = y_train

X_test = X_test.reshape((X_test.shape[0], -1))
X_test = X_test / 255

model = Model()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(Adam(learning_rate=0.01, beta1=0.9, beta2=0.9))
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)
