import numpy as np
import cv2
import math

####################################
def getmean(imm):
	
	M=imm.shape[0]
	N=imm.shape[1]

	s=0
	for i in range(M):
		for j in range(N):
			s+=imm[i,j]

	return s/(M*N)
	
################################
def getdev(imm1,imm2,u1,u2):
	row1=imm1.shape[0]
	col1=imm1.shape[1]

	s=0

	for i in range(row1):
		for j in range(col1):
			t1=imm1[i,j]-u1
			t2=imm2[i,j]-u2

			s+=t1*t2

	return s/(row1*col1)
##########################################

def getsig(imm,u):
	row=imm.shape[0]
	col=imm.shape[1]

	s=0

	for i in range(row):
		for j in range(col):

			s+=pow(imm[i,j]-u,2)

	return s/(row*col)




#############################################
def ssim(imm1,imm2):

	L=255
	c1=1/math.sqrt(L)
	c2=1/math.sqrt(L)

	u1=getmean(imm1)
	u2=getmean(imm2)

	dev=getdev(imm1,imm2,u1,u2)

	sx=getsig(imm1,u1)
	sy=getsig(imm2,u2)

	####################################

	k1=pow(c1*L,2)
	k2=pow(c2*L,2)

	f1=2*u1*u2+k1
	f2=2*dev+k2
	f3=pow(u1,2)+pow(u2,2)+k1
	f4=pow(sx,2)+pow(sy,2)+k2


	result = f1*f2/(f3*f4)


	return result


##################################################
############################################ 
im1=cv2.imread("img1.png",cv2.IMREAD_GRAYSCALE)
im2=cv2.imread("img2.png",cv2.IMREAD_GRAYSCALE)
im3=cv2.imread("img3.png",cv2.IMREAD_GRAYSCALE)

r1=ssim(im1,im2)
r2=ssim(im1,im3)

see1="ssim between img1 and img2 is {}".format(r1)
see2="ssim between img1 and img3 is {}".format(r2)

print(see1)
print(see2)

