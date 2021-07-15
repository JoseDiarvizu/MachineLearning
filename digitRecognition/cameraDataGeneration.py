import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.svm import SVC
import csv

df = pd.read_csv('/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/dataset6labels.csv',sep=",")
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
print('Número de muestras:',len(ytest))
print('Número de predicciones correctas:', np.sum(ytest==ypred))
print('macro-precision:',precision_score(ytest,ypred,average='macro'))
print('macro-recall:',recall_score(ytest,ypred,average='macro'))
print('macro-f1:',f1_score(ytest,ypred,average='macro'))


images_folder = "/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/fotos/3/"
label = "5"

def function(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(15,15),400)
    ret, imgT = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(imgT.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    cont = 0
    xData = []
    for rect in rectangles:
        datasetTestCamera = []
        if (rect[2] > 50 and rect[3]>50) and (rect[2] < 600 and rect[3]<600): 
            #print('img_shape',img.shape)
            imgn = img[ rect[1]:rect[1]+rect[3] , rect[0]:rect[0]+rect[2]]
            #cv2.imshow("Imagen nueva", imgn)
            #cv2.imwrite('CameraImages/img'+str(cont)+'.png',imgn)
            #imgn = cv2.resize(imgn,(100,100))
            #cv2.imwrite('CameraImages/img'+str(cont)+'.png',imgn)
            cont+=1

            ##################OBTENER HISTOGRAMA##################
            img_hsv = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)
            
            lower = np.array([0,0,0])
            upper = np.array([179,255,127])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper) 
            
            #print('img_hsv',img_hsv.shape)
            #nrows, ncols, nch = img_hsv.shape
            mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imshow("Imagen nueva", mask)  
            # Vectorizar la imagen
            rows,cols = mask.shape

            X=[]
            X.append(label)
            # #Add pixel one-by-one into data Array.
            for i in range(rows):
	            for j in range(cols):
	                k = mask[i,j]
	                if k>100:
	        	        k=1
	                else: 
	        	        k=0	
	                X.append(k)
            
            with open('/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/camera.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(X)
            
    return img


'''
cam = cv2.VideoCapture(0)
while True:
    val,img = cam.read()
    img = function(img)
    cv2.imshow("Imagen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

captura = cv2.VideoCapture('/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/5.mov')
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
######BILL RECOGNITION#########
