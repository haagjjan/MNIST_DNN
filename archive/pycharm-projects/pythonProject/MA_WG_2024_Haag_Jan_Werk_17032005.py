import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import activations
from keras import layers
from keras.layers import Dropout
from keras import backend as k


# Diese Funktion dient zur Überprüfung der Libraryversionen von Tensorflow, Keras und Numpy.
def versions():
    # Ich drucke die Versionen der jeweiligen Librarys, um mögliche falsche Versionen direkt zu sehen.
    print(tf.__version__)
    print(keras.__version__)
    print(np.__version__)


# Ich nutze eine Variable, um das Mnist-Dataset direkt abzurufen können.
mnist = tf.keras.datasets.mnist

# Nun übertrage ich die Daten des Mnist-Datasets auf 4 Variabeln.
# 2 Variabeln für die Daten für das Training. 2 Variabeln für die Daten für das Testing.
# die Variabeln Startend mit x_ haben die X-Werte der Daten. Die Variabeln Startend mit y_ haben die Y_Werte der Daten.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Nun normalisiere ich die X-Werte der 60'000 Zahlen des Mnist Dataset. Dies führt zur Umkehrung der Pixelwerte.
# So ist jeder Schwarze pixel nun Weiss.
# 0-255 --> 0-1, nur pixel normalisiert
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Die Folgende Funktion erstellt ein ANN Model.
# Hierfür werden keine zusätzlichen Daten benötigt.
def create_ann():
    # Nun erstelle ich ein Model, welches eine Sequentielle anordnung haben wird.
    # Eine Sequentielle Anordnung bedeutet, dass die Layer des ANN aneinander Führen.
    model = tf.keras.models.Sequential()
    # Jetzt füge ich den ersten Layer an das Modell.
    # Die Input Form ist eine Matrix an 28x28 Zahlenwerten zwischen 0-1.
    # Dabei handelt es sich um einen Flachgedückten Layer.
    # Dies bedeutet, dass die Input Form pixel für Pixel auseinandergenommen wird.
    # So besteht der Inputlayer aus 718 Neuronen.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # Jetzt füge ich dem Modell einen zweiten Layer an.
    # Der Layer besteht aus 120 Neuroenen.
    # Dabei ist jedes Neuron des Input Layers mit jedem Neuron des 1. Hidden Layer verbunden.
    # Die Aktivierungsfunktion der Neuronen ist jewiels die ReLU-Aktivierungsfunktion.
    model.add(layers.Dense(150, activation=activations.relu))
    # Nun führe ich einen Dropoutprozess nach dem 1. Hidden Layer ein.
    # Dabei wird 20% aller Neuronen temoporär abgeworfen.
    model.add(Dropout(0.2))
    # Jetzt füge ich dem Modell einen dritten Layer an.
    # Der Layer besteht aus 120 Neuroenen.
    # Dabei ist jedes Neuron des 1. Hidden Layers mit jedem Neuron des 2. Hidden Layer verbunden.
    # Die Aktivierungsfunktion der Neuronen ist jewiels die ReLU-Aktivierungsfunktion.
    model.add(layers.Dense(150, activation=activations.relu))
    # Nun führe ich einen Dropoutprozess nach dem 2. Hidden Layer ein.
    # Dabei wird 20% aller Neuronen temoporär abgeworfen.
    model.add(Dropout(0.2))
    # Nun füge ich dem bestehenden Modell einen Letzten Layer an.
    # Dieser Neuroen des 2. Hidden Layers ist mit jedem Neuron des Outputlayers verbunden.
    # Die Aktivierungsfunktion ist die Softmax-Aktivierungsfunction.
    model.add(layers.Dense(10, activation=activations.softmax))
    # Als letztes gebe ich das Model aus.
    return model


# In dieser Funktion wird das zuvor erstellte Model überarbeitet. Es werden einige Hyperparameter eingeführt.
# Hierfür wird das zuvor kreierte Model benötigt.
def compile_ann(model):
    # Nun führe ich einige Hyperparameter ein, und entscheide mich für eine Metrics.
    model.compile(optimizer="adam",                         # Ich wähle den Optimizer Adam Aus.
                  loss="sparse_categorical_crossentropy",   # Ich wähle die "sparse_categorical_crossentropy" Funktion.
                                                            # als lossfunction.
                  metrics=["accuracy"])                     # Als Metrics möchte ich die Genauigktiet.
    # Dies ist kein Hyperparameter.
    k.set_value(model.optimizer.learning_rate, 0.002)  # Hier verändere ich die Learningrate auf 0.002.
    # Als letztes gebe ich das Model aus.
    return model


# In dieser Funktion wird das zuvor erstellte Model Trainiert.
# Hierfür wird das zuvor kreierte Model benötigt.
def fit_ann(model):
    # Nun Trainiere ich das Modell mit den Trainingsdaten.
    # Hier lege ich die dauer des Trainings fest. Dies wird anhand der Epochenzahl bestummen.
    # Eine Epoche Besteht aush 1850 Trainings Daten.
    model.fit(x_train, y_train, epochs=1)
    # Als letztes gebe wird das Model ausgegeben.
    return model


# In dieser Funktion wird die Genauigkeit des ANN anhand der Testdaten getestet.
# Hierfür wird das zuvor kreierte Model benötigt.
def loss_accuracy(model):
    # Nach dem Trainieren, möchte ich das Model erneut Testen.
    # Das Model Evalueire wird an dem Testdatensatz getestet.
    # Nach der Evaluierung gebe ich den Loss und die Genauigkeit des Modells neu an.
    loss, accuracy = model.evaluate(x_test, y_test)
    # Nun speichere ich einen Antwortsatz kombiniert mit dem Loss in der Variable V_loss
    v_loss = ("This is the models loss: " + str(loss))
    # Nun speichere ich einen Antwortsatz kombiniert mit der Accuracy in der Variable V_accuracy
    v_accuracy = ("This is the models accuracy: " + str(accuracy))
    # Als letztes gebe ich die Werte für V_loss und V_accuracy aus.
    return v_loss, v_accuracy


# Abschliessend möchte man womöglich das Model an eigenen Zahlen testen.
# Hierfür ist diese Variable als erster Drittel des Pfads, der lokalen Datai vorgesehen.
locfile0 = r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit"
# Ebenfalls ist diese Variable als dritter Drittel des Pfads, der lokalen Datai vorgesehen.
locfile2 = ".png"


# Das Format der lokalen Datai besteht aus 3 Teilen.
    # 1. Teil: Ganzer Pfad bis zur Nummer vor dem Dataientypanhängsel
    # 2. Teil: Nummerierung der Dokumente (nicht die abgebildete Nummer im Dokument)
    # 3. Teil: Dataientyp Anhängsel
# bsp.:
    # 1. Teil: r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit"
    # 2. Teil: 189
    # 3. Teil: .png
    # So entsteht schlussendlich: r"\Users\Jan_MA\Downloads\Jan_MA\Digits_Example\Digit189.png"


# In dieser Funktion versucht das zuvor erstellte Model Lokal abgespeicherte Zahlen zu erkennen.
# Hierfür wird das zuvor kreierte Model benötigt.
# Ebenfalls müssen die Variablen locfile0 und locfile2 richtig angepasst sein.
def predict_input(model):
    # Dies ist der Startnummer des zweiten Teils lokalen Datai, so wird beinflsust, welche Zahl als erstes gesehen wird.
    image_number = 0
    # Die While-Schleife dient als Check. Nur falls eine Solche Datai tatsächlich exixtiert wird der Test gestartet.
    # Dies hat der Effekt, dass das Programm automatisch fortfährt,
    # nachdem es alle lokalen Daten im richtigen Format getestet hat.
    while os.path.isfile(locfile0 + str(image_number) + locfile2) == True:
        # Nun wird der erste Test gestartet.
        # Falls der Test nicht durchgeführt werden kann, wird der "escept:" Abschnitt durchgeführt.
        try:
            # Das Bild an dem zuvor festgelegten ort wird gelesen.
            # Hirfür werden alle 3 Teile des Pfads aneinandergesetzt.
            # Da das gelesene Bild automatisch im RGB Format ist,
            # wird mit den eckigen Klammern der RGB Wert für Blau und Grün gelöscht.
            # So hat man nur noch den RGB Wert Rot.
            # So hat nur noch die Form der Nummer relevanz. Die Farben spielen keine Rolle mehr.
            img = cv2.imread(locfile0 + str(image_number) + locfile2)[:, :, 1]
            # Die Liste, aus welcher das Bild besteht, wird umgedreht.
            # So wird jeder Pixel, der zuvor Schwarz war nun Weiss.
            img = np.invert(np.array([img]))
            # Nun wird dieses Bild durch das Modell gelesen durchgeschickt.
            # Mittels Forwardpropagation erhält man nun ein Resultat.
            prediction_img = model.predict(img)
            # Dieses Resultat wird Gedruckt.
            # Das Resultat gibt jedoch immer noch alle Werte der Outputnuronen aus.
            # Dies wird in einem Listenformat gemacht. der 1. Wert ist 0, der Letzte Wert ist 9.
            print(prediction_img)
            # Anschliessend wird die höchste Zahl der Outputneuronen genommen
            # Dann wird geschaut, welcher Vorhersagung dies entspricht.
            # Diese Vorhersagung wird dann gedruckt.
            print(f"This digit is probably a {np.argmax(prediction_img)}")
            # Um dem User die Lokale Datai zu zeigen, wird die Zahl jeweils noch abgebildet.
            plt.imshow(img[0], cmap=plt.cm.binary)
            # Nach Abbilden dieser Zahl, wird die lokale Datai User gezeigt.
            plt.show()
        # Wenn etwas im "try:" Abschnitt falsch läuft, wird dieser übersprungen,
        # und es wird der Programm Abschnitt "except:" gestartet
        except:
            # Da der Programmabschnitt "try:" übersprungen wurde, gab es einen fehler in diesem Programm abschnitt.
            # Da diese Fehler teilweise schwer zu erkennen wind, wird "Error!" gedruckt.
            print("Error!")
        finally:
            # Die Nummer des 2. Teil des Pfads wird erhöht.
            image_number += 1
            # Diese Nummer wird für mögliche überprüfungen gedruckt.
            print(image_number)


# Die Funktion "create_ann()" wird durchgeführt.
# Der ausgegebene Return "model" wird in der Variable "structure" gespeichert.
structure = create_ann()

# Die Funktion "compile_ann(model)" wird durchgeführt.
# Als Input für "model" wird die Variable "structure" genommen.
# Der ausgegebene Return "model" wird in der Variable "model" gespeichert.
model = compile_ann(structure)

# Die Funktion "fit_ann(model)" wird durchgeführt.
# Als Input für "model" wird die Variable "model" genommen.
# Der ausgegebene Return "model" wird in der Variable "model_fit" gespeichert.
model_fit = fit_ann(model)

# Die Funktion "fit_ann(model)" wird durchgeführt.
# Als Input für "model" wird die Variable "model_fit" genommen.
# Der ausgegebene Return "v_loss" und "v_accuracy" wird automatisch gedruckt.
loss_accuracy(model_fit)

# Die Funktion "predict_input(model_fit)" wird durchgeführt.
# Als Input für "model" wird die Variable "structure" genommen.
# Es kommt zu einigen automatisch gedruckten Ausgaben.
predict_input(model_fit)
