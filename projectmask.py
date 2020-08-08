# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:07:16 2020

@author: Welcome
"""


import pandas as pd #Pandas Library for dealing with Data Frames
import numpy as np # Numpy Library for dealing with multidimensional arrays
import cv2        # Opencv Library for detecting faces
from sklearn import metrics # scikit Library for evaluation
from sklearn.linear_model import LogisticRegression # Importing logistic regression model from sklearn Library
from sklearn.model_selection import train_test_split
data=pd.read_csv("DataFrame.csv")   #loading the DataFrame containing image data
data.head()
a=np.ones(100)  
a1=a.reshape(100,1)
y=pd.DataFrame(data=a1)
y.head()
b=np.zeros(100)
b1=b.reshape(100,1)
c=pd.DataFrame(data=b1)
y=y.append(c)
x=data
x.head(1)
d1=pd.read_csv("data1.csv")
x=x.append(d1)
f=np.ones(107)
f1=f.reshape(107,1)
f2=pd.DataFrame(data=f1)
y=y.append(f2)
d2=pd.read_csv("data2.csv")
x=x.append(d2)
g=np.ones(100)
g1=g.reshape(100,1)
g2=pd.DataFrame(data=g1)
y=y.append(g2)
h=np.zeros(200)
h1=h.reshape(200,1)
h2=pd.DataFrame(data=h1)
y=y.append(h2)
X = x.iloc[:, :12288].values    # Creating a Numpy array of DataFrame
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=4)  #Splitting the Training and Testing Data
logistic_regression = LogisticRegression()      
logistic_regression.fit(x_train,y_train)    # Fitting the model
y_pred=logistic_regression.predict(x_test)  #Predecting output for test Data
accuracy=metrics.accuracy_score(y_test,y_pred)  # Calculating the accuracy of the model
accuracy_percentage=100*accuracy
print(accuracy_percentage)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # Loading the haarcascade file Used for detecting the faces

cap = cv2.VideoCapture(0)   #Opening the system camera  *** If it doesn't work with '0' try with replace '1' in place of '0' ***

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)    # Getting the co-ordinates of the face

    for (z, y , w ,h) in faces:
        
        #roi_gray = gray[y:y+h, z:z+w]
        roi_color = img[y:y+h, z:z+w]
        im1=img[y:y+h,z:z+w]            # Getting the pixel values of Face only
        i1=cv2.resize(im1,(500,500))    
        im2=cv2.resize(i1,(64,64))      # Resizing the image into (64 X 64) pixels
        im3=im2.reshape(1,12288)
        df=pd.DataFrame(data=im3)       # Creating the data frame for image data
        pred=logistic_regression.predict(df)    # Calling the Logistic Regression model by giving image data as argument
        
        if pred[0]==1:          # If person is wearing mask predicted output is 1 
            cv2.rectangle(img, (z,y), (z+w, y+h), (0, 255, 0), 2)   # Drawing a Green color Rectangle around the face of the Person
            font=cv2.FONT_HERSHEY_SIMPLEX      # Giving the Font style
            cv2.putText(img,"Mask Detected",(z-10,y-10),font,0.5,(255,0,0),2,cv2.LINE_AA)   # Writing the text "mask detected" on the video 
        
        else:                   # If person is not wearing mask predicted output is 0 
            cv2.rectangle(img, (z,y), (z+w, y+h), (0, 0, 255), 2)       # Drawing a Red color Rectangle around the face of the Person
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,"No Mask Detected",(z-10,y-10),font,0.5,(255,0,0),2,cv2.LINE_AA)    # Writing the text "no mask detected" on the video

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):       # When q is pressed in the keyboard the camera window will be closed 
        break

cap.release()
cv2.destroyAllWindows()