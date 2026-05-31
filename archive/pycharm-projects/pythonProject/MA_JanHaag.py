import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import activations
from keras import layers
from keras.layers import Dropout
from keras import backend as K

def versions():
    print(tf.__version__)
    print(keras.__version__)
    print(np.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def create_ann():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    #model.add(Dropout(0.2))
    model.add(layers.Dense(120, activation=activations.relu))
    model.add(Dropout(0.5))
    model.add(layers.Dense(120, activation=activations.relu))
    model.add(Dropout(0.5))
    model.add(layers.Dense(10, activation=activations.softmax))
    return model

def compile_ann(model):
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    K.set_value(model.optimizer.learning_rate, 0.002)
    return model

def fit_ann(model):
    model.fit(x_train, y_train, epochs=1)
    return model

def loss_accuracy(model):
    loss, accuracy = model.evaluate(x_test, y_test)
    V_loss = ("This is the models loss: " + str(loss))
    V_accuracy = ("This is the models accuracy: " + str(accuracy))
    return V_loss, V_accuracy

locfile0 = r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit"
locfile2 = ".png"

def predict_input(model):
    image_number = 0
    while os.path.isfile(locfile0 + str(image_number) + locfile2) == True:
        try:
            img = cv2.imread(locfile0 + str(image_number) + locfile2)[:,:,0]
            img = np.invert(np.array([img]))
            prediction_img = model.predict(img)
            print(prediction_img)
            print(f"This digit is probably a {np.argmax(prediction_img)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")
        finally:
            image_number += 1
            print(image_number)

structure = create_ann()
model = compile_ann(structure)
model_fit = fit_ann(model)
loss_accuracy(model_fit)
predict_input(model_fit)

