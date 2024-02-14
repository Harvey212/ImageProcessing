import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def hist_equal(img):
	row=img.shape[0]
	col=img.shape[1]
	ch=img.shape[2]

	result=np.copy(img)
	his=[]
	pp=[]

	for k in range(0,ch):
		histr = calHist(img,k)
		sum_histr=int(sum(histr))

		pdf=[0]*len(histr)

		for i in range(0,len(histr)):
			pdf[i]=int((histr[i]))/sum_histr

		upper=len(pdf)-1
		cdf=0

		trans={}

		CDF=[]
		CDF.append(0)
		PDF=[]
		bef=0
		aff=0

		for i in range(0,len(pdf)):
			if i !=0:
				checkbef=int(upper*cdf)

			cdf=cdf+pdf[i]
			CDF.append(cdf)
			check=int(upper*cdf)
			if(i==0):
				checkbef=check

			trans[i]=check
			if check!=checkbef:
				diff=CDF[aff]-CDF[bef]
				PDF.append(diff)
				bef=aff
			aff=aff+1

		Last=1-sum(PDF)
		PDF.append(Last)
		
		for i in range(0,row):
			for j in range(0,col):
				val=img[i,j,k]
				search=trans[val]
				result[i,j,k]=search
		his.append(trans)
		pp.append(PDF)
	
	final=result
	

	return final,his,pp





def show():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def barpdf(img,his,ch,ss):
	sparsex=list(set(his[ch].values()))

	dic={}
	for i in range(0,len(sparsex)):
		dic[sparsex[i]]=i

	vec=[0]*len(sparsex)

	spli=img[:,:,ch]
	row=spli.shape[0]
	col=spli.shape[1]

	for i in range(0,row):
		for j in range(0,col):
			check=spli[i][j]
			vec[dic[check]]+=1

	summ=sum(vec)
	transpdf=[x/summ for x in vec]

	plt.figure(ss)
	plt.bar(sparsex, transpdf)
	plt.show()



def oribar(img,ch,ss):
	histr = calHist(img,ch)
	sum_histr=int(sum(histr))

	pdf=[0]*len(histr)

	for i in range(0,len(histr)):
		pdf[i]=int(histr[i])/sum_histr

	xb=[i for i in range(0,256)]

	plt.figure(ss)
	plt.bar(xb, pdf)
	plt.show()


def calHist(imm,channel):
	test=[0]*256

	row=imm.shape[0]
	col=imm.shape[1]

	for i in range(0,row):
		for j in range(0,col):
			pos=image[i][j][channel]
			test[pos]+=1

	return test



image=cv2.imread('einstein-low-contrast.tif')




res,his1,p1=hist_equal(image)

hmerge=np.hstack((image,res))

cv2.imshow('(1)Left Picture: Original image. Right Picture: Histogram equalization',hmerge)
show()


####################################################################
oribar(image,0,'Before histogram equalization')
barpdf(res,his1,0,'After histogram equalization')




