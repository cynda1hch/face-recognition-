import os
import cv2
from tqdm import tqdm
from random import shuffle
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models,datasets,layers
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

all_data = []
DIRECTORY = 'data'
for category in os.listdir(DIRECTORY):
    category_path = os.path.join(DIRECTORY, category)
    for img in tqdm(os.listdir(category_path)):
        img_path = os.path.join(category_path, img)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (224, 224))
        all_data.append([resized_image, category])

shuffle(all_data)

images_alone = []
labels_alone = []
for vector in all_data:
    images_alone.append(vector[0])
    labels_alone.append(vector[1])

x = np.array(images_alone)
y = np.array(labels_alone)

x = x/255.0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
Y = to_categorical(y, num_classes=4)

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
model.save('detection_model')


history=model.fit(x, Y, batch_size=32, epochs=5, validation_split=0.1)

image = tf.expand_dims(x_test[0], axis=0)
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
cv2.imshow('test', x_test[0])
print(y_test[0])

cv2.waitKey(0)
cv2.destroyAllWindows()
