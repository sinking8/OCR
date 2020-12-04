import pytesseract
import cv2
import numpy as np 
import imutils

class Detect:

	img = None

	def __init__(self,img):
		pytesseract.pytesseract.tesseract_cmd = r'.\project\Tesseract-OCR\tesseract'
		self.img = img

	def detect_text(self):

		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		gray  =cv2.GaussianBlur(gray,(3,3),0,0)
		edged = cv2.Canny(gray, 30, 200) #Perform Edge detection

		text  =  self.detect_license_plate(edged,gray)
		return text

	def detect_license_plate(self,edged,gray):

		contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
		screenCnt = None

		for c in contours:
		    # approximate the contour
		    peri = cv2.arcLength(c, True)
		    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
		    # if our approximated contour has four points, then
		    # we can assume that we have found our screen
		    if len(approx) == 4:
		        screenCnt = approx
		        break

		# Masking the part other than the number plate
		mask = np.zeros(gray.shape,np.uint8)
		new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
		new_image = cv2.bitwise_and(self.img,self.img,mask=mask)

		# Now crop
		(x, y) = np.where(mask == 255)
		(topx, topy) = (np.min(x), np.min(y))
		(bottomx, bottomy) = (np.max(x), np.max(y))
		Cropped = gray[topx:bottomx+1, topy:bottomy+1]

		return(pytesseract.image_to_string(Cropped))