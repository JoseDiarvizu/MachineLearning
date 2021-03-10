import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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


datasetBilletesX = []
datasetBilletesY = []
tipoBillete = ["MX020N_","MX050N_","MX100N_","MX200N_","MX500N_"]
letraBillete = ['a','b','c','d']
numeroBillete = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20"]

###LEER BILLETES DE 20#####

for i in range(0,1):
    for j in numeroBillete:
        for k in letraBillete:
            #print(tipoBillete[i]+j+k)
            if j ==  "01":
                continue
            elif j == "02" and k == "a":
                continue
            else:
                #print(tipoBillete[i]+j+k)
                img = cv2.imread("billetes/"+tipoBillete[i]+j+k+".jpg")
                img_bgr = img
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                #print('img',img.shape)


                lower = np.array([85, 30, 50])
                upper = np.array([120, 200, 255])
                lower = np.array([0, 50, 50])
                upper = np.array([180, 255, 255])

                #Encontrar los pixeles de la imagen azules
                mask = cv2.inRange(img_hsv, lower, upper)

                '''
                plt.figure()
                plt.subplot(1,2,1)
                plt.title('rgb')
                plt.imshow( img_rgb )
                plt.subplot(1,2,2)
                plt.title('mask')
                plt.imshow( mask,cmap='gray' )
                plt.show()
                '''
                
                #print('img_hsv',img_hsv.shape)
                nrows, ncols, nch = img_hsv.shape

                # Vectorizar la imagen
                Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
                #print('Ximg_hsv',Ximg_hsv.shape)

                # Hue | Value | Saturation
                # 24  |  92   |  156
                # 100 |  234  |  234

                # Obtener hue
                hue = Ximg_hsv[:,0]
                '''
                plt.figure()
                plt.hist(hue)
                plt.show()
                '''
                histograma = np.histogram(hue,bins=10,range=[0,180])[0]
                histograma = histograma / np.sum(histograma)
                histograma = np.round(histograma,2)
                datasetBilletesX.append(histograma)
                datasetBilletesY.append(20)

###LEER BILLETES DE 20#####


###LEER BILLETES DE 50#####

for i in range(1,2):
    for j in numeroBillete:
        for k in letraBillete:
            #print(tipoBillete[i]+j+k)
            img = cv2.imread("billetes/"+tipoBillete[i]+j+k+".jpg")
            img_bgr = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)


            lower = np.array([130, 20, 50])
            upper = np.array([180, 250, 255])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper)

            '''
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('rgb')
            plt.imshow( img_rgb )
            plt.subplot(1,2,2)
            plt.title('mask')
            plt.imshow( mask,cmap='gray' )
            plt.show()
            '''
                
            #print('img_hsv',img_hsv.shape)
            nrows, ncols, nch = img_hsv.shape

            # Vectorizar la imagen
            Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
            #print('Ximg_hsv',Ximg_hsv.shape)

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            # Obtener hue
            hue = Ximg_hsv[:,0]
            '''
            plt.figure()
            plt.hist(hue)
            plt.show()
            '''
            histograma = np.histogram(hue,bins=10,range=[0,180])[0]
            histograma = histograma / np.sum(histograma)
            histograma = np.round(histograma,2)
            datasetBilletesX.append(histograma)
            datasetBilletesY.append(50)

###LEER BILLETES DE 50#####

###LEER BILLETES DE 100#####
for i in range(2,3):
    for j in numeroBillete:
        for k in letraBillete:
            #print(tipoBillete[i]+j+k)
            img = cv2.imread("billetes/"+tipoBillete[i]+j+k+".jpg")
            img_bgr = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)


            lower = np.array([0, 50, 50])
            upper = np.array([20, 200, 255])
            lower = np.array([0, 50, 50])
            upper = np.array([180, 255, 255])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper)

            '''
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('rgb')
            plt.imshow( img_rgb )
            plt.subplot(1,2,2)
            plt.title('mask')
            plt.imshow( mask,cmap='gray' )
            plt.show()
            '''
                
            #print('img_hsv',img_hsv.shape)
            nrows, ncols, nch = img_hsv.shape

            # Vectorizar la imagen
            Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
            #print('Ximg_hsv',Ximg_hsv.shape)

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            # Obtener hue
            hue = Ximg_hsv[:,0]
            '''
            plt.figure()
            plt.hist(hue)
            plt.show()
            '''
            histograma = np.histogram(hue,bins=10,range=[0,180])[0]
            histograma = histograma / np.sum(histograma)
            histograma = np.round(histograma,2)
            datasetBilletesX.append(histograma)
            datasetBilletesY.append(100)


###LEER BILLETES DE 100#####


###LEER BILLETES DE 200#####

for i in range(3,4):
    for j in numeroBillete:
        for k in letraBillete:
            #print(tipoBillete[i]+j+k)
            img = cv2.imread("billetes/"+tipoBillete[i]+j+k+".jpg")
            img_bgr = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)


            lower = np.array([30, 10, 30])
            upper = np.array([80, 255, 255])
            lower = np.array([0, 50, 50])
            upper = np.array([180, 255, 255])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper)

            '''
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('rgb')
            plt.imshow( img_rgb )
            plt.subplot(1,2,2)
            plt.title('mask')
            plt.imshow( mask,cmap='gray' )
            plt.show()
            '''
                
            #print('img_hsv',img_hsv.shape)
            nrows, ncols, nch = img_hsv.shape

            # Vectorizar la imagen
            Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
            #print('Ximg_hsv',Ximg_hsv.shape)

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            # Obtener hue
            hue = Ximg_hsv[:,0]
            '''
            plt.figure()
            plt.hist(hue)
            plt.show()
            '''
            histograma = np.histogram(hue,bins=10,range=[0,180])[0]
            histograma = histograma / np.sum(histograma)
            histograma = np.round(histograma,2)
            datasetBilletesX.append(histograma)
            datasetBilletesY.append(200)

###LEER BILLETES DE 200#####

###LEER BILLETES DE 500#####

for i in range(4,5):
    for j in numeroBillete:
        for k in letraBillete:
            #print(tipoBillete[i]+j+k)
            img = cv2.imread("billetes/"+tipoBillete[i]+j+k+".jpg")
            img_bgr = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)


            lower = np.array([0, 0, 30])
            upper = np.array([30, 50, 255])
            lower = np.array([0, 50, 50])
            upper = np.array([180, 255, 255])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper)

            '''
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('rgb')
            plt.imshow( img_rgb )
            plt.subplot(1,2,2)
            plt.title('mask')
            plt.imshow( mask,cmap='gray' )
            plt.show()
            '''
                
            #print('img_hsv',img_hsv.shape)
            nrows, ncols, nch = img_hsv.shape

            # Vectorizar la imagen
            Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
            #print('Ximg_hsv',Ximg_hsv.shape)

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            # Obtener hue
            hue = Ximg_hsv[:,0]
            '''
            plt.figure()
            plt.hist(hue)
            plt.show()
            '''
            histograma = np.histogram(hue,bins=10,range=[0,180])[0]
            histograma = histograma / np.sum(histograma)
            histograma = np.round(histograma,2)
            datasetBilletesX.append(histograma)
            datasetBilletesY.append(500)

###LEER BILLETES DE 500#####


###MODELO################

datasetBilletesX = np.array(datasetBilletesX)
datasetBilletesY = np.array(datasetBilletesY)
Xtrain, Xtest, ytrain, ytest = train_test_split(datasetBilletesX, datasetBilletesY, test_size=0.10, random_state=42)


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
###MODELO################



######BILL RECOGNITION#########


def function(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),400)
    ret, imgT = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(imgT.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    cont = 0
    for rect in rectangles:
        datasetTestCamera = []
        if (rect[2] > 50 and rect[3]>50) and (rect[2] < 3000 and rect[3]<3000): 
            #print('img_shape',img.shape)
            imgn = img[ rect[1]:rect[1]+rect[3] , rect[0]:rect[0]+rect[2]]
            cv2.imshow("Imagen nueva", imgn)
            #cv2.imwrite('CameraImages/img'+str(cont)+'.png',imgn)
            #imgn = cv2.resize(imgn,(100,100))
            #cv2.imwrite('CameraImages/img'+str(cont)+'.png',imgn)
            cont+=1

            ##################OBTENER HISTOGRAMA##################
            img_hsv = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)

            #print('img',img.shape)


            lower = np.array([0, 25, 50])
            upper = np.array([180, 255, 255])

            #Encontrar los pixeles de la imagen azules
            mask = cv2.inRange(img_hsv, lower, upper)    
            #print('img_hsv',img_hsv.shape)
            nrows, ncols, nch = img_hsv.shape

            # Vectorizar la imagen
            Ximg_hsv = np.reshape( img_hsv, (nrows*ncols,3) )
            #print('Ximg_hsv',Ximg_hsv.shape)

            # Hue | Value | Saturation
            # 24  |  92   |  156
            # 100 |  234  |  234

            # Obtener hue
            hue = Ximg_hsv[:,0]
            histograma = np.histogram(hue,bins=10,range=[0,180])[0]
            histograma = histograma / np.sum(histograma)
            histograma = np.round(histograma,2)
            datasetTestCamera.append(histograma)
            datasetTestCamera = np.array(datasetTestCamera)
            ypred = model.predict(datasetTestCamera)
            #print("##Predicción del Modelo##", ypred)
            ##################OBTENER HISTOGRAMA##################

            #cv2.imwrite('CameraImages/img'+str(cont)+'.png',mask)
            

            # Clasificar la imagen


            cv2.rectangle(img, (rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (0,255,0),2)
            cv2.putText(img, str(ypred), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 3, cv2.LINE_AA)
 
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
captura = cv2.VideoCapture('billsVideo.mp4')
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
