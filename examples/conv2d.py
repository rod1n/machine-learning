from nn import Model, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))
X_train = X_train / 255
y_train = y_train

X_test = X_test.reshape((X_test.shape[0], -1))
X_test = X_test / 255

model = Model()
model.add(Conv2D(4, (2, 2), activation='sigmoid', input_shape=(1, 28, 28)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(keep_prob=0.5))
model.add(Conv2D(4, (2, 2), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile()
model.fit(X_train, y_train, epochs=10, learning_rate=0.001, batch_size=1000, verbose=True)