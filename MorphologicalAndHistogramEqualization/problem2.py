from __future__ import print_function
import numpy as np
import cv2 

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
	invGamma = 1.0 / gamma
	one_invGamma=1-invGamma
	#table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	table = np.array([(i** invGamma)*(255**one_invGamma) for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



img=cv2.imread('aerialview-washedout.tif')
adjusted = adjust_gamma(img, gamma=0.25)


scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(adjusted, dim, interpolation = cv2.INTER_AREA)

combine=np.hstack((resized1,resized2))
cv2.imshow('gamma correction',combine)
cv2.waitKey(0)







