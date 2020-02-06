from nn import Model
from nn.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

# model = Model()
# model.add(Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=(1, 28, 28)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(4, (2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))
# model.compile(Adam(learning_rate=0.01, beta1=0.9, beta2=0.9))
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)

import matplotlib.pyplot as plt
from nn.initializers import *

rnd = Model()
rnd.add(Dense(16, 'sigmoid', input_shape=(784,), weight_initializer=RandomNormal()))
rnd.add(Dense(10, 'softmax', weight_initializer=RandomNormal()))
rnd.compile(Adam(learning_rate=0.01, beta1=0.9, beta2=0.9))
rnd.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)

glorot = Model()
glorot.add(Dense(16, 'sigmoid', input_shape=(784,), weight_initializer=GlorotNormal()))
glorot.add(Dense(10, 'softmax', weight_initializer=GlorotNormal()))
glorot.compile(Adam(learning_rate=0.01, beta1=0.9, beta2=0.9))
glorot.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)

plt.plot(rnd.scores['acc'])
plt.plot(glorot.scores['acc'])
plt.legend(['rnd', 'glorot'])
plt.show()