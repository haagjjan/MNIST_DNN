image_number = 1
while os.path.isfile(r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit" + str(image_number) +".png") == True:
    try:
        img = cv2.imread(r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit" + str(image_number) +".png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
        print(image_number)

import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def invert_lib(image_file, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    cmap_val = None if not gray_scale else 'gray'
    image_i = cv2.bitwise_not(image_src)

    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

        ax1.axis("off")
        ax1.title.set_text('Original')

        ax2.axis("off")
        ax2.title.set_text("Inverted")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_i, cmap=cmap_val)
        return True
    return image_i

image_number2 = 1
#fileloc = r"\Users\Jan_MA\Downloads\Jan_MA\Portable_Network_Graphics\Figure_" + str(image_number2) +".png"
fileloc0 = r"\Users\Jan_MA\Downloads\Jan_MA\Portable_Network_Graphics\Figure_"
fileloc1 = ".png"

while os.path.isfile(fileloc0 + str(image_number2) + fileloc1) == True:
    loc = fileloc0 + str(image_number2) + fileloc1
    print(loc)
    try:
        img = cv2.imread("r" + loc)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        print(image_number2)
        image_number2 += 1
        print(loc)
