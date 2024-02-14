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

#convert to PIL format #background image
new_im = Image.fromarray(newimg.astype(np.uint8))

#import the foreground image
img2=cv2.imread('graveler.bmp')
#convert to PIL format
img2=Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#add a alpha channel for transparency
new_img2=img2.convert('RGBA')

#get pixel info
datas=new_img2.getdata()

#make the white background into transparency
newData=[]

for item in datas:
	if item[0]==255 and item[1]==255 and item[2]==255:
		newData.append((255,255,255,0))
	else:
		newData.append(item)

#modify the original image with the new data with transparent background
new_img2.putdata(newData)

#overlay the image at (200,250)
new_im.paste(new_img2,(200,250),new_img2)
new_im.show()









