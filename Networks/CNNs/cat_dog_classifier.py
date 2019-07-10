import plaidml.keras
plaidml.keras.install_backend()

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time

#load in the data
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

X= X/255.0 #normalizing the pixel data which has a max value of 255

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

            model = Sequential()

            #First conv-ReLU-Pooling block
            model.add(Conv2D(layer_size, (3,3), input_shape = (100, 100, 1), activation="relu"))
            model.add(MaxPooling2D(pool_size = (2, 2)))

            #Second conv-ReLU-Pooling block
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3), activation="relu"))
                model.add(MaxPooling2D(pool_size = (2, 2)))

            model.add(Flatten())
            #Fully connected layer
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dropout(0.3))

            #Output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss = "binary_crossentropy", optimizer="adam", metrics = ['accuracy'])

            model.fit(X, y, batch_size = 32, validation_split=0.1, epochs=10, callbacks = [tensorboard])

model.save('64x3-CNN.model')
