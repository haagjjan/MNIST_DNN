import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import activations
from keras import layers
from keras.layers import Dropout


print(tf.__version__)
print(keras.__version__)
print(np.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(Dropout(0.2))
model.add(layers.Dense(120, activation=activations.relu))
model.add(Dropout(0.2))
model.add(layers.Dense(120, activation=activations.relu))
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation=activations.softmax))
#model.add(tf.keras.layers.Dense(128, activation=activations.relu))
#model.add(tf.keras.layers.Dense(128)) #, activition='relu'
#model.add(tf.keras.layers.Dense(128)) # , activition='relu'
#model.add(tf.keras.layers.Dense(10)) #, activition='softmax'

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs=1)
model.save("HandwrittenDigitRecognition")

#modelsavefile = tf.keras.models.load_model("hanwritten.model")

loss, accuracy = model.evaluate(x_test, y_test)
print("This is the models loss: " + str(loss))
print("This is the models accuracy: " + str(accuracy))

#digit_folder_Path = "\Users\Jan_MA\Downloads\Jan_MA\Digits_Example"
#Digitexample = "\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit4.png"

import json

#def read_this(image_file, gray_scale=False):
#    image_src = cv2.imread(image_file)
#    if gray_scale:
#        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
#    else:
#        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
#    return image_src

#def invert_lib(image_file, with_plot=False, gray_scale=False):
#    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
#    cmap_val = None if not gray_scale else 'gray'
#    image_i = cv2.bitwise_not(image_src)
#
#    if with_plot:
#        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
#
#        ax1.axis("off")
#        ax1.title.set_text('Original')
#
#        ax2.axis("off")
#        ax2.title.set_text("Inverted")
#
#        ax1.imshow(image_src, cmap=cmap_val)
#        ax2.imshow(image_i, cmap=cmap_val)
#        return True
#    return image_i

locfile0 =  r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit"
locfile1 = ".png"

image_number = 0
while os.path.isfile(r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit" + str(image_number) +".png") == True:
    try:
        img = cv2.imread(r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit" + str(image_number) +".png")[:,:,0]
        img = np.invert(np.array([img]))
        #invertedfile1 = invert_lib(image_file=locfile0 + str(0) + locfile1, with_plot=True)
        #prediction1 = invertedfile1
        prediction1 = model.predict(img)
        #invertedfile2 = invert_lib(image_file=locfile0 + "-1" + locfile1, with_plot=True)
        #prediction2 = invertedfile2
        #prediction2 = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction1)}")
        #print(f"This digit is probably a {np.argmax(prediction2)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
        print(image_number)

plot_model(self.model, to_file = path, show_shapes = True)












