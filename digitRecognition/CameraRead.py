
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.svm import SVC

df = pd.read_csv('camera.csv',sep=",")
X = df.values[0:10000,1:]
Y = df.values[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.10, random_state=42)
print('X',X.shape)
print('Xtrain',Xtrain.shape)
print('Xtest',Xtest.shape)



model = SVC(random_state=42)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)
print(Xtrain.shape, Xtest.shape, len(ytrain), len(ytest))
print('accuracy_score',accuracy_score(ytest,ypred))
print('Samples:',len(ytest))
print('Correct predictions:', np.sum(ytest==ypred))
print('macro-precision:',precision_score(ytest,ypred,average='macro'))
print('macro-recall:',recall_score(ytest,ypred,average='macro'))
print('macro-f1:',f1_score(ytest,ypred,average='macro'))


def function(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(15,15),400)
    ret, imgT = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(imgT.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    cont = 0
    for rect in rectangles:
        datasetTestCamera = []
        if (rect[2] > 50 and rect[3]>50) and (rect[2] < 600 and rect[3]<600): 
            imgn = img[ rect[1]:rect[1]+rect[3] , rect[0]:rect[0]+rect[2]]
            cont+=1
            img_hsv = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)

            
            lower = np.array([0,0,0])
            upper = np.array([179,255,127])
            mask = cv2.inRange(img_hsv, lower, upper) 
            
            mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imshow("Imagen nueva", mask)   
            # Vectorizar la imagen
            rows,cols = mask.shape

            X=[]
            # #Add pixel one-by-one into data Array.
            for i in range(rows):
	            for j in range(cols):
	                k = mask[i,j]
	                if k>100:
	        	        k=1
	                else: 
	        	        k=0	
	                X.append(k)
            
            

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            
            ypred = model.predict([X])
            print("-----------Imagen camara: ",ypred,"-----------")
            print(X)
            cv2.imwrite('CameraImages/img'+str(cont)+'.png',mask)            

            # Clasificar la imagen
            cv2.rectangle(img, (rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (0,255,0),2)
            cv2.putText(img, str(ypred), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 3, cv2.LINE_AA)
 
    return img


cam = cv2.VideoCapture(0)
while True:
    val,img = cam.read()
    img = function(img)
    cv2.imshow("Imagen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
captura = cv2.VideoCapture('videoBilletes4.MOV')
while (captura.isOpened()):
  ret, img = captura.read()

  img = function(img)

  if ret == True:
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break
captura.release()
cv2.destroyAllWindows()
'''
######DIGIT RECOGNITION#########
