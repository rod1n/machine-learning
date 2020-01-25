from nn import Model
from nn.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=(1, 28, 28)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile()
model.fit(X_train, y_train, epochs=10, learning_rate=0.1, batch_size=32, verbose=True)
