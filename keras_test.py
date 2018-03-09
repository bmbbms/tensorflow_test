
# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(len(X_train), -1)
y_train = np_utils.to_categorical(y_train, 10)

x_test = X_test.reshape(len(X_test), -1)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(input_dim=28*28, output_dim=500))
model.add(Activation("sigmoid"))

model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))

model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss="mse",
              optimizer=SGD(lr=0.1),
              metrics=['accuracy']
              )

model.fit(x_train, y_train, batch_size=100, nb_epoch=20)

score = model.evaluate(x_test, y_test)
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])

result = model.predict(x_test)