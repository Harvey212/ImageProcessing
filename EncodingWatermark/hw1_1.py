import numpy as np
import cv2
from PIL import Image


img1=cv2.imread('laptop_left.png')
img2=cv2.imread('laptop_right.png')

vis = np.concatenate((img1, img2), axis=1)
cv2.imshow('image',vis)
cv2.waitKey()
cv2.imwrite('out.png', vis)