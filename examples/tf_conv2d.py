from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow_core.python.keras.losses import categorical_crossentropy
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3)
# X_train = X_train.reshape((X_train.shape[0], -1))
X_train = X_train / 255
y_train = y_train
y_train = keras.utils.to_categorical(y_train, 10)


X_test = np.expand_dims(X_test, axis=3)
# X_test = X_test.reshape((X_train.shape[0], -1))
X_test = X_test / 255
y_test = keras.utils.to_categorical(y_test, 10)



model = Sequential()
model.add(Conv2D(16, (3, 3), activation='sigmoid' , input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
# model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
result = model.fit(X_train, y_train, epochs=10, batch_size=32)


score = model.evaluate(X_train, y_train)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
