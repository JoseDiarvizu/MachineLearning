import pyscreenshot as ImageGrab
import cv2
import time

images_folder = "/Users/arvizu/Downloads/UP/SEASON6/MachineLearning/Python/digitRecog/fotos/3/"

for i in range (0,45):
	time.sleep(7)
	im = ImageGrab.grab(bbox=(80, 80, 208, 208)) # X1,Y1,X2,Y2
	print ("saved....",i)
	im.save(images_folder+str(i)+'.png')
	img = cv2.imread(images_folder+str(i)+'.png')
	cv2.namedWindow("Result")
	cv2.imshow("Result", img)
	print("clear screen now and redraw now...")
	cv2.waitKey(1000)
	