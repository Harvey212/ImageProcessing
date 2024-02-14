import numpy as np
import cv2
from PIL import Image

img=cv2.imread('lena_flipped.bmp')
newimg=np.zeros((img.shape[0],img.shape[1],img.shape[2]))

for i in range(0,(img.shape[0]-1)):
	rowp=img.shape[0]-1-i
	for j in range(0,(img.shape[1]-1)):
		columnp=img.shape[1]-1-j
		#because r,g,b->b,g,r
		newimg[rowp][columnp][2]=img[i][j][0]
		newimg[rowp][columnp][1]=img[i][j][1]
		newimg[rowp][columnp][0]=img[i][j][2]


new_im = Image.fromarray(newimg.astype(np.uint8))
new_im.show()


#convert PIL format to opencv
#new_im = np.array(new_im)
#new_im = new_im[:, :, ::-1].copy() 
#cv2.imshow('image',new_im)
#cv2.waitKey()
