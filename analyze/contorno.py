import cv2 
import numpy as np 
  
# Let's load a simple image with 3 black squares 
image = cv2.imread('104010.jpg') 
  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Source', gray)

ret,img = cv2.threshold(gray, 20, 255, 0)

cv2.imshow('Threshold', img)

edged = cv2.Canny(gray, 100, 200)
  
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Canny Edges After Contouring', edged) 

cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
  
cv2.imshow('Contours', image)

cv2.waitKey(0) 
cv2.destroyAllWindows() 