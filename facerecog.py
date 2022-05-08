import os
import cv2
fc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cv2.namedWindow("test")
img = None
false = 0

DIRECTORY = "data"
CATEGORIES = ['eya', 'ouma', 'senda', 'sywar']

for i in CATEGORIES:
    for j in os.listdir(os.path.join(DIRECTORY, i)):
        image = cv2.imread(os.path.join(DIRECTORY, i, j), 0)
        face = fc.detectMultiScale(image, 1.1, 4)
        for (x, y, w, h) in face:
            img = image[y:y + h, x:w + x]
        if img is not None:
            cv2.imwrite(os.path.join('data', i+'_'+j), img)
            print(i)
        else:
            false += 1
print(false)