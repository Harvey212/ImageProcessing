import numpy as np 
import cv2
from PIL import Image
from matplotlib import pyplot as plt 
from numpy import median
import math

def medfilter(imm):
	row=imm.shape[0]
	col=imm.shape[1]


	newim=np.zeros((row,col))

	windowsizee=5

	for i in range(0,row):
		for j in range(0,col):
			centerrow=i
			centercol=j
			span=int((windowsizee-1)/2)
		
			up=centerrow-span
			down=centerrow+span
			left=centercol-span
			right=centercol+span
			if up<0:
				up=0
			if down>(row-1):
				down=row-1
			if left<0:
				left=0
			if right>(col-1):
				right=col-1

			test=imm[up:(down+1),left:(right+1)]
			candy=[]
		
			rr=test.shape[0]
			cc=test.shape[1]

			summ=[]

			for k in range(0,rr):
				for m in range(0,cc):
					if (test[k][m]!=255) and (test[k][m]!=0):
						candy.append(test[k][m])
					else:
						summ.append(test[k][m])

			if len(candy)!=0:
				newim[i][j]=median(candy)
			else:
				newim[i][j]=np.mean(summ)

	return newim



def gaufilter(imm2):
	row2=imm2.shape[0]
	col2=imm2.shape[1]


	newim2=np.zeros((row2,col2))

	windowsizee2=5

	#kernel=[[0.0232,0.0338,0.0383,0.0338,0.0232],
	#[0.0338,0.0492,0.0558,0.0492,0.0338],
	#[0.0383,0.0558,0.0632,0.0558,0.0383],
	#[0.0338,0.0492,0.0558,0.0492,0.0338],
	#[0.0232,0.0338,0.0383,0.0338,0.0232]]

	for i in range(0,row2):
		for j in range(0,col2):
			centerrow2=i
			centercol2=j
			span2=int((windowsizee2-1)/2)
		
			up2=centerrow2-span2
			down2=centerrow2+span2
			left2=centercol2-span2
			right2=centercol2+span2
			if up2<0:
				up2=0
			if down2>(row2-1):
				down2=row2-1
			if left2<0:
				left2=0
			if right2>(col2-1):
				right2=col2-1

			var=((1/0.0632)**2)/(2*math.pi)

			summ2=0
			sumG=0

			tcheck=0
			tt=0
			count=0


			for k in range(up2,(down2+1)):
				for m in range(left2,(right2+1)):
					if (imm2[k][m]!=255) and (imm2[k][m]!=0):
						dis=(k-centerrow2)**2+(m-centercol2)**2
						G=(1/(math.sqrt((2*math.pi*var))))*math.exp(-(dis)/(2*var))
						ff=G*imm2[k][m]
						summ2=summ2+ff
						sumG=sumG+G
						tcheck=1
					else:
						tt=tt+imm2[k][m]
						count=count+1
					

			if tcheck==1:
				newim2[i][j]=summ2/sumG
			else:
				newim2[i][j]=tt/count
			

	return newim2


###########################################################

img=cv2.imread('bab_noise.bmp',0)
img2=cv2.imread('peppers_noise.bmp',0)

#####################################################
p1=medfilter(img)

combine1=np.hstack((img,p1))

new_im1 = Image.fromarray(combine1)
new_im1.show(title='median filter bab_noise.bmp')
##########################################################
p2=medfilter(img2)

combine2=np.hstack((img2,p2))

new_im2 = Image.fromarray(combine2)
new_im2.show(title='median filter peppers_noise.bmp')

####################################################################
p3=gaufilter(img)

combine3=np.hstack((img,p3))

new_im3 = Image.fromarray(combine3)
new_im3.show(title='Gaussian filter bab_noise.bmp')

################################################
p4=gaufilter(img2)

combine4=np.hstack((img2,p4))

new_im4 = Image.fromarray(combine4)
new_im4.show(title='Gaussian filter peppers_noise.bmp')


