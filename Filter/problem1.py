import numpy as np  
import cv2
from PIL import Image
from matplotlib import pyplot as plt 

def filt(mag):
	test=[]
	row=mag.shape[0]
	col=mag.shape[1]

	for i in range(0,row):
		for j in range(0,col):
			test.append(mag[i][j])

	test.sort(reverse=True)
	ind=int((row*col)/4)
	bench=test[(ind-1)]

	mask=np.zeros((row, col), np.uint8)

	for i in range(0,row):
		for j in range(0,col):
			val=mag[i][j]
			if val>=bench:
				mask[i][j]=1

	return mask


def MSE(img1, img2):
	squared_diff = (img1 -img2)**2
	summed = np.sum(squared_diff)
	num_pix = img1.shape[0] * img1.shape[1]
	err = summed / num_pix
	
	return err



img=cv2.imread('bridge.jpg',0).astype('int16')


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))


plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


MASK=filt(magnitude_spectrum)
res=np.multiply(fshift,MASK)

f_ishift = np.fft.ifftshift(res)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

new_im = Image.fromarray(img_back)
new_im.show(title="Image A")
cv2.waitKey(0)

#############################################################


rowstart=0
rowend=15

colstart=0
colend=15

marchi=0
marchj=0

ff=0

for i in range(0,16):
	if i==0:
		marchi=0
	else:
		marchi=16

	rowstart=rowstart+marchi
	rowend=rowend+marchi
	colstart=0
	colend=15

	colarr=0

	for j in range(0,16):
		if j==0:
			marchj=0
		else:
			marchj=16

		colstart=colstart+marchj
		colend=colend+marchj
		block=magnitude_spectrum[rowstart:(rowend+1),colstart:(colend+1)]
		blockmask=filt(block)
		if j==0:
			colarr=blockmask
		else:
			colarr=np.hstack((colarr,blockmask))

	if i==0:
		ff=colarr
	else:
		ff=np.vstack((ff,colarr))


res2=np.multiply(fshift,ff)

f_ishift2 = np.fft.ifftshift(res2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)


new_im2 = Image.fromarray(img_back2)
new_im2.show(title="Image B")
cv2.waitKey(0)

###########################################################


mm = cv2.pyrDown(img)

fg = np.fft.fft2(mm)
fshift3 = np.fft.fftshift(fg)

res3= np.pad(fshift3, pad_width=64, mode='constant', constant_values=0)


f_ishift3 = np.fft.ifftshift(res3)
img_back3 = np.fft.ifft2(f_ishift3)
img_back3 = np.abs(img_back3)


new_im3 = Image.fromarray(img_back3)
new_im3.show(title="Image C")
cv2.waitKey(0)



########################################################################
D_A=MSE(img,img_back)

D_B=MSE(img,img_back2)

D_C=MSE(img,img_back3)


print(f"Mean square error between image and image A: {D_A}")

print(f"Mean square error between image and image B: {D_B}")

print(f"Mean square error between image and image C: {D_C}")


