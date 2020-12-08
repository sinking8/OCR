from tensorflow import keras
import cv2
import numpy as np 
import imutils

class Detect:

	img = None

	def __init__(self,img):
		self.img = img
		self.detect_model = keras.models.load_model('./project/detect.h5',compile=False)
		self.lenet_model  = keras.models.load_model('./project/Lenet.h5',compile=False)

	def detect_text(self):
		text  =  self.detect_license_plate()
		return text

	def detect_license_plate(self):

		try:

			#Generating key value pairs
			ch  = 'A'
			preds ={};
			for i in range(1,27):preds[i] = chr(ord(ch) + i)
			
			#Applying GaussianBlur and performing Edge Detection
			gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
			blurred = cv2.GaussianBlur(gray, (5, 5), 0)
			edged = cv2.Canny(blurred, 50, 200, 255)

			cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,gray)

			#Detecting Contours
			contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			code = ''


			if(len(contours) !=0):
			    for c in contours:

			        x, y, w, h = cv2.boundingRect(c)     
			        area = cv2.contourArea(c)
			        
			        if(area>100):        
			            img = self.img[y:y+h,x:x+w]
			            img = cv2.resize(img,(32,32))
			            img = np.reshape(img,(1,32,32,3))
			            
			            pred  = np.argmax(self.lenet_model.predict(img), axis = -1)[0]
			            
			            if(pred>9):
			                pred  = preds[pred]
			            code  = code + ' ' + str(pred)
			return code

		except:
			return False