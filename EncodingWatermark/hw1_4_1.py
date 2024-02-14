import numpy as np
import cv2
from PIL import Image

def encode(im,im2):
	em=im2.shape
	im_c_=np.copy(im)

	for i in range(0,(em[0]-1)):
		for j in range(0,(em[1]-1)):
			c0=format(int(im_c_[i][j][0]),'08b')
			c1=format(int(im_c_[i][j][1]),'08b')
			c2=format(int(im_c_[i][j][2]),'08b')

			v0=format(int(im2[i][j][0]),'08b')
			v1=format(int(im2[i][j][1]),'08b')
			v2=format(int(im2[i][j][2]),'08b')

			r0=0
			r1=0
			r2=0

			for k in range(0,3):
				c0check=int(c0[k])
				c1check=int(c1[k])
				c2check=int(c2[k])
				power=7-k
				if(c0check==1):
					r0=r0+pow(2,power)
				if(c1check==1):
					r1=r1+pow(2,power)
				if(c2check==1):
					r2=r2+pow(2,power)

			for m in range(0,3):
				v0check=int(v0[m])
				v1check=int(v1[m])
				v2check=int(v2[m])
				power2=3-m
				if(v0check==1):
					r0=r0+pow(2,power2)
				if(v1check==1):
					r1=r1+pow(2,power2)
				if(v2check==1):
					r2=r2+pow(2,power2)

			im_c_[i][j][0]=r0
			im_c_[i][j][1]=r1
			im_c_[i][j][2]=r2

	return im_c_

def flip(imm):
	newimg=np.zeros((imm.shape[0],imm.shape[1],imm.shape[2]))

	for i in range(0,(imm.shape[0]-1)):
		rowp=imm.shape[0]-1-i
		for j in range(0,(imm.shape[1]-1)):
			columnp=imm.shape[1]-1-j
			#because r,g,b->b,g,r
			newimg[rowp][columnp][2]=imm[i][j][0]
			newimg[rowp][columnp][1]=imm[i][j][1]
			newimg[rowp][columnp][0]=imm[i][j][2]

	return newimg


def decode(imd,rd,ld):
	newd1=np.zeros((imd.shape[0],imd.shape[1],imd.shape[2]))
	newd2=np.zeros((rd,ld,3))

	rowd=0
	cold=0
	for i in range(0,(imd.shape[0]-1)):
		rowd=rowd+1
		cold=0
		for j in range(0,(imd.shape[1]-1)):
			if((rowd<rd) and (cold<ld)):
				split0=imd[i][j][0]
				split1=imd[i][j][1]
				split2=imd[i][j][2]

				origin0=int(split0/16)
				sepa0=split0%16

				origin0=origin0*16
				sepa0=sepa0*16

				origin1=int(split1/16)
				sepa1=split1%16

				origin1=origin1*16
				sepa1=sepa1*16

				origin2=int(split2/16)
				sepa2=split2%16

				origin2=origin2*16
				sepa2=sepa2*16

				newd1[i][j][0]=origin0
				newd1[i][j][1]=origin1
				newd1[i][j][2]=origin2

				newd2[i][j][0]=sepa0
				newd2[i][j][1]=sepa1
				newd2[i][j][2]=sepa2
			else:
				newd1[i][j][0]=imd[i][j][0]
				newd1[i][j][1]=imd[i][j][1]
				newd1[i][j][2]=imd[i][j][2]
				
			cold=cold+1

	return newd1,newd2

def encode2(im_,im2_):
	im_c=np.copy(im_)
	em_=im2_.shape
	for i in range(0,(em_[0]-1)):
		for j in range(0,(em_[1]-1)):
			inse0=(im2_[i][j][0])/1000
			inse1=(im2_[i][j][1])/1000
			inse2=(im2_[i][j][2])/1000
			im_c[i][j][0]=im_c[i][j][0]+inse0
			im_c[i][j][1]=im_c[i][j][1]+inse1
			im_c[i][j][2]=im_c[i][j][2]+inse2
	return im_c

def decode2(imd_,rd_,ld_):
	newd1_=np.zeros((imd_.shape[0],imd_.shape[1],imd_.shape[2]))
	newd2_=np.zeros((rd_,ld_,3))

	rowd_=0
	cold_=0
	for i in range(0,(imd_.shape[0]-1)):
		rowd_=rowd_+1
		cold_=0
		for j in range(0,(imd_.shape[1]-1)):
			if((rowd_<rd_) and (cold_<ld_)):
				split0_=(imd_[i][j][0])*1000
				split1_=(imd_[i][j][1])*1000
				split2_=(imd_[i][j][2])*1000

				origin0_=int(split0_/1000)
				sepa0_=split0_%1000

				origin1_=int(split1_/1000)
				sepa1_=split1_%1000

				origin2_=int(split2_/1000)
				sepa2_=split2_%1000

				newd1_[i][j][0]=origin0_
				newd1_[i][j][1]=origin1_
				newd1_[i][j][2]=origin2_

				newd2_[i][j][0]=sepa0_
				newd2_[i][j][1]=sepa1_
				newd2_[i][j][2]=sepa2_
			else:
				newd1_[i][j][0]=imd_[i][j][0]
				newd1_[i][j][1]=imd_[i][j][1]
				newd1_[i][j][2]=imd_[i][j][2]
				
			cold_=cold_+1

	return newd1_,newd2_

img=cv2.imread('lena_flipped.bmp')
lp=flip(img)

img2=cv2.imread('graveler.bmp')

#method1
combine1=encode(lp,img2)
combine1_e = Image.fromarray(combine1.astype(np.uint8))
E1 = cv2.cvtColor(np.array(combine1_e), cv2.COLOR_RGB2BGR)
cv2.imshow('method1_encode',E1)
cv2.waitKey(0)

ori,water=decode(combine1,img2.shape[0],img2.shape[1])

ori_d = Image.fromarray(ori.astype(np.uint8))
D1 = cv2.cvtColor(np.array(ori_d), cv2.COLOR_RGB2BGR)
cv2.imshow('method1_decode_original',D1)
cv2.waitKey(0)

water_d = Image.fromarray(water.astype(np.uint8))
D2 = cv2.cvtColor(np.array(water_d), cv2.COLOR_RGB2BGR)
cv2.imshow('method1_decode_watermark',D2)
cv2.waitKey(0)

#method2
combine2=encode2(lp,img2)
combine2_e = Image.fromarray(combine2.astype(np.uint8))
E2 = cv2.cvtColor(np.array(combine2_e), cv2.COLOR_RGB2BGR)
cv2.imshow('method2_encode',E2)
cv2.waitKey(0)

ori_,water_=decode2(combine2,img2.shape[0],img2.shape[1])

ori_d_ = Image.fromarray(ori_.astype(np.uint8))
D1_ = cv2.cvtColor(np.array(ori_d_), cv2.COLOR_RGB2BGR)
cv2.imshow('method2_decode_original',D1_)
cv2.waitKey(0)

water_d_ = Image.fromarray(water_.astype(np.uint8))
D2_ = cv2.cvtColor(np.array(water_d_), cv2.COLOR_RGB2BGR)
cv2.imshow('method2_decode_watermark',D2_)
cv2.waitKey(0)

cv2.destroyAllWindows()
############################################################
#img2=Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#new_img2=img2.convert('RGBA')
#datas=new_img2.getdata()

#newData=[]

#for item in datas:
#	if item[0]==255 and item[1]==255 and item[2]==255:
#		newData.append((255,255,255,0))
#	else:
#		c=list(item)
#		c[3]=0
#		x=tuple(c)
#		newData.append(x)


#new_img2.putdata(newData)
#new_im = Image.fromarray(lp.astype(np.uint8))
#new_im.paste(new_img2,(0,0),new_img2)

#new_im.show()
#############################################################








