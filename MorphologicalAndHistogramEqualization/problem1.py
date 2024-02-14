import numpy as np 
import cv2 


image = cv2.imread('text-broken.tif')
kernel = np.ones((3,3),np.uint8)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Repairment', closing)
cv2.waitKey(0)


gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Hollow', gradient)
cv2.waitKey(0)
