import keras
import tensorflow.keras
from keras.datasets import mnist,cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i  in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
plt.show()

num_classes = 10

y_train = to_categorical(y_train, num_classes)

y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_test.shape[0], 'test samples')

model=Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

opt = SGD(learning_rate=0.01, momentum=0.9, decay=0.0002, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

classifer= model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
print("the model has sucessfully trained")
model.save('cifar10.h5')
print("the model has been saved")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import cv2

img = cv2.imread("C:/Users/PC/Desktop/image.jpg")

def load_image(filename, size=(32, 32)):
    img=load_image(filename,target_size=size)
    img=img_to_array(img)
    img=np.expand_dims(img, axis=0)
    return img

def run_example():
    model = load_model('cifar10.h5')
    img = load_image('C:/Users/PC/Desktop/image.jpg')
    result = model.predict(img)
    if result[0][0] > 0.5:
        print('cat')
    else:
        print('dog')
run_example()
   
   







   