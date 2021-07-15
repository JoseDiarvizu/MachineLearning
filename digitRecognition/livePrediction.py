import joblib,cv2
import numpy as np
import pyscreenshot as ImageGrab
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/dataset6labels.csv',sep=",")
X = df.values[:1000,1:]
Y = df.values[:1000,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.10, random_state=42)
print('X',X.shape)
print('Xtrain',Xtrain.shape)
print('Xtest',Xtest.shape)


'''
model = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=2,
                    learning_rate_init=.1, verbose=True)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)
print(Xtrain.shape, Xtest.shape, len(ytrain), len(ytest))
print('accuracy_score',accuracy_score(ytest,ypred))
print('Número de muestras:',len(ytest))
print('Número de predicciones correctas:', np.sum(ytest==ypred))
print('macro-precision:',precision_score(ytest,ypred,average='macro'))
print('macro-recall:',recall_score(ytest,ypred,average='macro'))
print('macro-f1:',f1_score(ytest,ypred,average='macro'))
'''

model = SVC(random_state=42)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

print('Acc:',accuracy_score(ytest,ypred))
print('Macro F1:',f1_score(ytest,ypred,average='macro'))
print( confusion_matrix(ytest,ypred) )


images_folder = "/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/fotos"

for i in range (0,100):
	
	
	img = ImageGrab.grab(bbox=(80, 80, 208, 208)) # X1,Y1,X2,Y2
	img.save("test_orig.png")
	im = cv2.imread("test_orig.png")
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

	# Threshold the image
	ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)


	roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

	cv2.imwrite("segmented.png", roi)
	

	rows,cols = roi.shape

	X=[]

	# #Add pixel one-by-one into data Array.
	for i in range(rows):
	    for j in range(cols):
	        k = roi[i,j]
	        if k>100:
	        	k=1
	        else: 
	        	k=0	
	        X.append(k)
	
	predictions = model.predict([X])      
	print("Prediction: ", predictions[0])
	cv2.putText(im, "Prediction is: "+str(predictions[0]), (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	
	#cv2.startWindowThread()
	cv2.namedWindow("Result")
	cv2.imshow("Result", im)
	cv2.waitKey(2000)

	#scaling = MinMaxScaler(feature_range=(-1, 1)).fit([X])

	#X = scaling.transform([X])	
	#time.sleep(4)