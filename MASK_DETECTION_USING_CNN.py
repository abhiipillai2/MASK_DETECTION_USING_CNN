import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')
with_mask.shape

X = np.r_[with_mask,without_mask]
X.shape[0]

labels = np.zeros(X.shape[0])

labels[500 :] = 1.0

x_train,x_test,y_train,y_test = train_test_split(X,labels,test_size=0.21)

X_train = x_train / 255
X_test = x_test / 255
X_train.shape
print(f"test data {X_test.shape}")

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

face_matrix = np.random.randn(1,50,50,3)
y_predicted = cnn.predict(np.expand_dims(X_test[0],0))


cnn.evaluate(X_test,y_test)

haar_data = cv2.CascadeClassifier('data.xml')

capture = cv2.VideoCapture(0)
data = []
while True:
    flag,img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img,(x,y),(w+x,h+y),(255,0,255),4)
            face_matrix = img[y:y+h,x:x+w,:]
            face_matrix = cv2.resize(face_matrix,(256,256))
            y_predicted = cnn.predict(np.expand_dims(face_matrix,0))
            
            out = int(np.argmax(y_predicted))
            if out == 0:
                print ('Mask')
            elif out == 1:
                print ('No- Mask')
            #print(y_predicted)
        cv2.imshow('Result',img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()