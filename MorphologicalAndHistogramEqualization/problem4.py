import numpy as np
import cv2

def hist_equal(img):
	row=img.shape[0]
	col=img.shape[1]
	ch=img.shape[2]


	result=np.copy(img)

	for k in range(0,ch):
		histr=calHist(img,k,7)

		cc,uu=calCDF(histr)

		tt=transform(cc,uu)
	
		for i in range(0,row):
			for j in range(0,col):
				val=img[i,j,k]
				search=tt[val]
				result[i,j,k]=search
		
	
	final=result
	

	return final





def show():
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def calHist(imm,channel,windowsize):
	rr=imm.shape[0]
	coll=imm.shape[1]

	test=np.zeros((256,256))	#row: center #col: outer
	
	width=int(windowsize/2)
	height=int(windowsize/2)

	for i in range(0,rr):
		for j in range(0,coll):
			upper=i-height
			lower=i+height
			left=j-width
			right=j+width

			if(upper<=0):
				upper=0
			if(lower>=rr):
				lower=rr-1
			if(left<=0):
				left=0
			if(right>=coll):
				right=coll-1
			
			center=imm[i][j][channel]

			for m in range(upper,(lower+1)):
				for n in range(left,(right+1)):
					pixel=imm[m][n][channel]
					test[center][pixel]=test[center][pixel]+1



	return test


def calCDF(pdf):
	

	cdf=[0]*256
	cdfu=[0]*256
	##################################################
	summtotal=0
	summrow=0
	summcol=0

	Usummtotal=0
	Usummrow=0
	Usummcol=0
	####################################################
	

	for k in range(0,256):
		for i in range(0,(k+1)):
			summrow=summrow+pdf[i][k]
			Usummrow=Usummrow+(1/(256*256))
		for j in range(0,(k+1)):
			summcol=summcol+pdf[k][j]
			Usummcol=Usummcol+(1/(256*256))

		summtotal=summtotal+summrow+summcol-pdf[k][k]
		Usummtotal=Usummtotal+Usummrow+Usummcol-(1/(256*256))

		cdf[k]=summtotal
		cdfu[k]=Usummtotal
		summrow=0
		summcol=0
		Usummrow=0
		Usummcol=0
	##########################################
	maxx=cdf[255]
	cdf=cdf/maxx

	return cdf,cdfu

def transform(CDF,CDFU):
	trans={}
	index=0


	for i in range(0,len(CDFU)):
		bench=CDFU[i]
		
		while CDF[index]<bench:
			trans[index]=i
			index=index+1

	while index<len(CDFU):
		trans[index]=len(CDFU)-1
		index=index+1

	return trans






image=cv2.imread('einstein-low-contrast.tif')


res=hist_equal(image)

hmerge=np.hstack((image,res))
cv2.imshow('(1)Left Picture: Original image. Right Picture: Histogram equalization',hmerge)
show()






