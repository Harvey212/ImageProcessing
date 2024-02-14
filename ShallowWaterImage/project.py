import os
import numpy as np
import cv2
import math
from skimage.color import rgb2hsv,hsv2rgb
from skimage.color import rgb2lab, lab2rgb
import statistics 
from matplotlib import pyplot as plt


######################################################

def stretching(img):
    height = img.shape[0]  #row of image
    width = img.shape[1]    #col of image
    for k in range(0, 3):
        #Max_channel  = np.max(img[:,:,k])
        #Min_channel  = np.min(img[:,:,k])


        length = height * width
        R_rray = (np.copy(img[:,:,k])).flatten()
        R_rray.sort()

        ###################################################
        mm=statistics.mode(R_rray)
        LL=list(R_rray)
        mindex=LL.index(mm)

        Min_channel = int(R_rray[int(mindex*0.1/ 100)])
        Max_channel = int(R_rray[-int( (length-mindex)*0.1/100 )])

        ####################################################
        #Min_channel = int(R_rray[int(length / 100)])
        #Max_channel = int(R_rray[-int(length / 100)])
        #############################################################
        modd=statistics.mode(R_rray)

        omin=int((1-0.655)*modd)
        omax=255


        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - Min_channel) * (omax - omin) / (Max_channel - Min_channel)+ omin  
    return img


##############################################################

def cal_equalisation(img,ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = []
    totalave=0

    for i in range(3):
        avg = np.mean(img[:,:,i])    ##
        avg_RGB.append(avg)
        totalave=totalave+avg

    totalave=totalave/3

    avg_RGB = 128/np.array(avg_RGB)  ##128/mean
    ratio = avg_RGB

    for i in range(0,2):
        img[:,:,i] = cal_equalisation(img[:,:,i],ratio[i])
    return img



########################################################

e = math.e

def global_Stretching_ab(a,height, width):
    array_Global_histogram_stretching_L = np.zeros((height, width), 'float64')
    for i in range(0, height):
        for j in range(0, width):
                p_out = a[i][j] * (1.3 ** (1 - math.fabs(a[i][j] / 128)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)



############################################################3
def global_stretching(img_L,height, width):
    length = height * width
    R_rray = (np.copy(img_L)).flatten()
    R_rray.sort()
    #print('R_rray',R_rray)
    I_min = int(R_rray[int(length / 1000)])
    I_max = int(R_rray[-int(length / 1000)])
    #print('I_min',I_min)
    #print('I_max',I_max)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 100
            else:
                p_out = int((img_L[i][j] - I_min) * ((100) / (I_max - I_min)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)


#############################################################

def  LABStretching(sceneRadiance):


    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    height = sceneRadiance.shape[0]
    width = sceneRadiance.shape[1]

    img_lab = rgb2lab(sceneRadiance)
    L, a, b = cv2.split(img_lab)

    img_L_stretching = global_stretching(L, height, width)
    img_a_stretching = global_Stretching_ab(a, height, width)
    img_b_stretching = global_Stretching_ab(b, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = img_L_stretching
    labArray[:, :, 1] = img_a_stretching
    labArray[:, :, 2] = img_b_stretching
    img_rgb = lab2rgb(labArray) * 255



    return img_rgb


###################################################################

def MSE(img1, img2):
    squared_diff = (img1 -img2)**2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]
    err = summed / num_pix
    
    return err




################################################################
imgg = cv2.imread('imm1.jpg')

#colour equalization
sceneRadiance = RGB_equalisation(imgg)

#RGHS
sceneRadiance = stretching(sceneRadiance)

#L*A*B* stretching
sceneRadiance = LABStretching(sceneRadiance)

value1 = MSE(imgg, sceneRadiance) 

print(f"Mean square error between images before and after processing: {value1}")




###############################################################

beforered=imgg[0]
beforegreen=imgg[1]
beforeblue=imgg[2]


plt.hist(beforered.ravel(),256,[0,256]);plt.title('red channel before processing');plt.show()
plt.hist(beforegreen.ravel(),256,[0,256]);plt.title('green channel before processing');plt.show()
plt.hist(beforeblue.ravel(),256,[0,256]);plt.title('blue channel before processing');plt.show()



##############################################################################
red=sceneRadiance[0]
green=sceneRadiance[1]
blue=sceneRadiance[2]


plt.hist(red.ravel(),256,[0,256]);plt.title('red channel after processing');plt.show()
plt.hist(green.ravel(),256,[0,256]);plt.title('green channel after processing');plt.show()
plt.hist(blue.ravel(),256,[0,256]);plt.title('blue channel after processing');plt.show()


cv2.imwrite('result1.jpg', sceneRadiance) 

