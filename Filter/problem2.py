import numpy as np 
import cv2
from PIL import Image
from matplotlib import pyplot as plt 


def LevelA(imgg,windowsizee,m,k):
	rr=imgg.shape[0]
	cc=imgg.shape[1]
	A1=0
	A2=0
	centerrow=m
	centercol=k
	condition=False
	span=int((windowsizee-1)/2)
	up=centerrow-span
	down=centerrow+span
	left=centercol-span
	right=centercol+span
	if up<0:
		up=0
	if down>(rr-1):
		down=rr-1
	if left<0:
		left=0
	if right>(cc-1):
		right=cc-1

	test=imgg[up:(down+1),left:(right+1)]
	med=np.median(test)
	maxx=np.amax(test)
	minn=np.amin(test)

	A1=med-minn
	A2=med-maxx

	if A1>0 and A2<0:
		condition=True

	return condition,med,maxx,minn




img=cv2.imread('Fig0514(a).tif',0).astype('int16')



row=img.shape[0]
col=img.shape[1]

newim=np.zeros((row,col))

for i in range(0,row):
	for j in range(0,col):
		
		windowsize=3
		check=False
		final=0
		center=img[i][j]
		B1=0
		B2=0

		for k in range(0,3):
			check,medd,maxxx,minnn=LevelA(img,windowsize,i,j)
			if check==True:
				break
			else:
				windowsize=windowsize+2


		if check==False:
			final=center
		else:
			B1=center-minnn
			B2=center-maxxx

			if B1>0 and B2<0:
				final=center
			else:
				final=medd

		newim[i][j]=final




new_im = Image.fromarray(newim)
new_im.show()






