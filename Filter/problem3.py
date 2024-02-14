import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt 


def repeatGaussian(blur,nn):

	for i in range(0,nn):
		blur = cv2.GaussianBlur(blur,(3,3),0)	

	f = np.fft.fft2(blur)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))

	nn=nn+1
	NAME = "Repeat {times} times (Gaussian filter)".format(times=str(nn))
	plt.subplot(121),plt.imshow(blur, cmap = 'gray')
	plt.title(NAME), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()

def repeatMedian(blur2,nn2):

	for i in range(0,nn2):
		blur2 = cv2.medianBlur(blur2,5)

	f2 = np.fft.fft2(blur2)
	fshift2 = np.fft.fftshift(f2)
	magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

	nn2=nn2+1
	NAME2 = "Repeat {times} times (Median filter)".format(times=str(nn2))
	plt.subplot(121),plt.imshow(blur2, cmap = 'gray')
	plt.title(NAME2), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(magnitude_spectrum2, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()




img=cv2.imread('bridge.jpg',0).astype('int16')
oneblur = cv2.GaussianBlur(img,(3,3),0)

#repeatGaussian(oneblur,9)


img2=cv2.imread('Fig0514(a).tif',0).astype('int16')
oneblur2 = cv2.medianBlur(img2,5)

repeatMedian(oneblur2,9)

