import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def hist_equal(img,mode):
	row=img.shape[0]
	col=img.shape[1]
	ch=img.shape[2]

	result=np.copy(img)
	his=[]
	pp=[]
	if mode==0:
		for k in range(0,ch):
			histr = cv2.calcHist([img],[k],None,[256],[0,256])
			sum_histr=int((sum(histr))[0])

			pdf=[0]*histr.shape[0]

			for i in range(0,histr.shape[0]):
				pdf[i]=int((histr[i])[0])/sum_histr

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
	else:
		if mode==1:
			imm= cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
			cc=2
		if mode==2:
			imm= cv2.cvtColor(result, cv2.COLOR_BGR2YCR_CB)
			cc=0

		histr = cv2.calcHist([imm],[cc],None,[256],[0,256])
		sum_histr=int((sum(histr))[0])

		pdf=[0]*histr.shape[0]

		for i in range(0,histr.shape[0]):
			pdf[i]=int((histr[i])[0])/sum_histr

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
				val=imm[i,j,cc]
				search=trans[val]
				imm[i,j,cc]=search

		final=imm
		his.append(trans)
		pp.append(PDF)

	return final,his,pp


def cv_hist(img,mode):
	if mode==1:
		result=np.copy(img)
		imm= cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(imm)
		v=cv2.equalizeHist(v)
		res = cv2.merge([h,s,v])
	if mode==0:
		result=np.copy(img)
		b,g,r = cv2.split(result)
		b=cv2.equalizeHist(b)	
		g=cv2.equalizeHist(g)
		r=cv2.equalizeHist(r)
		res = cv2.merge([b,g,r])
	if mode==2:
		result=np.copy(img)
		imm= cv2.cvtColor(result, cv2.COLOR_BGR2YCR_CB)
		y,cr,cb = cv2.split(imm)
		y=cv2.equalizeHist(y)
		res = cv2.merge([y,cr,cb])
	return res


def show():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def barpdf(img,his,ch,ss,mode):
	if mode==0:
		sparsex=list(set(his[ch].values()))
	else:
		sparsex=list(set(his[0].values()))

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
	histr = cv2.calcHist([img],[ch],None,[256],[0,256])
	sum_histr=int((sum(histr))[0])

	pdf=[0]*histr.shape[0]

	for i in range(0,histr.shape[0]):
		pdf[i]=int((histr[i])[0])/sum_histr

	xb=[i for i in range(0,256)]

	plt.figure(ss)
	plt.bar(xb, pdf)
	plt.show()


img=cv2.imread('mp2.jpg')

res,his1,p1=hist_equal(img,0)
res2=cv_hist(img,0)

hmerge = np.hstack((res, res2))
cv2.imshow('(1)Left Picture:my function implementation. Right Picture:opencv implementation',hmerge)
show()


###############################################################
img2=cv2.imread('mp2a.jpg')

res3,his2,p2=hist_equal(img2,0)
res3_=cv_hist(img2,0)

hmerge = np.hstack((res3, res3_))
cv2.imshow('(2)(a)R,G,B channels separately: Left Picture:my function implementation. Right Picture:opencv implementation',hmerge)
show()

#############################################################
res4,his3,p3=hist_equal(img2,1)
res4_=cv_hist(img2,1)

hmerge = np.hstack((res4, res4_))
cv2.imshow('(2)(b)V channel of HSV representation: Left Picture:my function implementation. Right Picture:opencv implementation',hmerge)
show()
###########################################################


res5,his4,p4=hist_equal(img2,2)
res5_=cv_hist(img2,2)

hmerge = np.hstack((res5, res5_))
cv2.imshow('(2)(c)Y channel of YCbCr representation: Left Picture:my function implementation. Right Picture:opencv implementation',hmerge)
show()


####################################################################
oribar(img,0,'(1) Blue channel before histogram equalization')
barpdf(res,his1,0,'(1) Blue channel after histogram equalization',0)

oribar(img,1,'(1) Green channel before histogram equalization')
barpdf(res,his1,1,'(1) Green channel after histogram equalization',0)

oribar(img,2,'(1) Red channel before histogram equalization')
barpdf(res,his1,2,'(1) Red channel after histogram equalization',0)




####################################################################
oribar(img2,0,'(2)(a) Blue channel before histogram equalization')
barpdf(res3,his2,0,'(2)(a) Blue channel after histogram equalization',0)

oribar(img2,1,'(2)(a) Green channel before histogram equalization')
barpdf(res3,his2,1,'(2)(a) Green channel after histogram equalization',0)

oribar(img2,2,'(2)(a) Red channel before histogram equalization')
barpdf(res3,his2,2,'(2)(a) Red channel after histogram equalization',0)



####################################################################
HSV= cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
oribar(HSV,2,'(2)(b) V channel of HSV representation before histogram equalization')
barpdf(res4,his3,2,'(2)(b) V channel of HSV representation after histogram equalization',1)

###############################################################################
YCB= cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
oribar(YCB,0,'(2)(c)Y channel of YCbCr representation before histogram equalization')
barpdf(res5,his4,0,'(2)(c)Y channel of YCbCr representation after histogram equalization',1)

