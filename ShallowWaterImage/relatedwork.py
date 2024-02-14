import math
import os

import numpy as np
import cv2
from skimage.color import rgb2hsv,hsv2rgb
from matplotlib import pyplot as plt

#############################################################################
def RGB_equalisation(img,height,width):

    img = np.float32(img)
    b, g, r = cv2.split(img)
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)

    All_avg = np.array((r_avg,g_avg,b_avg))
    All_max = np.max(All_avg)
    All_min = np.min(All_avg)
    All_median = np.median(All_avg)
    A = All_median/All_min
    B = All_median/All_max

    if (All_min == r_avg):
        r = r * A
    if (All_min == g_avg):
        g = g * A
    if (All_min == b_avg):
        b = b * A

    if (All_max == r_avg):
        r = r * B
    if (All_max == g_avg):
        g = g * B
    if (All_max == b_avg):
        b = b * B


    sceneRadiance = np.zeros((height, width, 3), 'float64')
    sceneRadiance[:, :, 0] = b
    sceneRadiance[:, :, 1] = g
    sceneRadiance[:, :, 2] = r
    sceneRadiance = np.clip(sceneRadiance, 0, 255)


    return sceneRadiance



#####################################################################
def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel  = np.max(img[:,:,k])
        Min_channel  = np.min(img[:,:,k])
        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel)+ 0
    return img



#################################################################################
def global_stretching(img_L,height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(img_L[i][j])
    R_rray.sort()
    I_min = R_rray[int(length / 100)]
    I_max = R_rray[-int(length / 100)]
    # print('I_min',I_min)
    # print('I_max',I_max)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = p_out
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = p_out
            else:
                p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)



#####################################################################################
def  HSVStretching(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    img_hsv[:, :, 1] = global_stretching(img_hsv[:, :, 1], height, width)
    img_hsv[:, :, 2] = global_stretching(img_hsv[:, :, 2], height, width)
    img_rgb = hsv2rgb(img_hsv) * 255

    return img_rgb



#####################################################################3

e = np.e
esp = 2.2204e-16



class NodeLower(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)



class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

def rayleighStrLower(nodes, height, width,lower_Position):
    alpha = 0.4
    selectedRange = [0, 255]
    NumPixel = np.zeros(256)
    temp = np.zeros(256)
    for i in range(0, lower_Position):
        # print('nodes[i].value',type(nodes[i].value))
        NumPixel[nodes[i].value] = NumPixel[nodes[i].value] + 1
    ProbPixel = NumPixel / lower_Position
    CumuPixel = np.cumsum(ProbPixel)
    # print('CumuPixel',CumuPixel)

    valSpread = selectedRange[1] - selectedRange[0]
    hconst = 2 * alpha ** 2
    vmax = 1 - e ** (-1 / hconst)
    val = vmax * (CumuPixel)
    val = np.array(val)

    for i in range(256):
        if (val[i] >= 1):
            val[i] = val[i] - esp
    for i in range(256):
        temp[i] = np.sqrt(-hconst * math.log((1 - val[i]), e))
        normalization = temp[i] * valSpread
        if(normalization > 255):
            CumuPixel[i] = 255
        else:
            CumuPixel[i] = normalization
    for i in range(0, lower_Position):
        nodes[i].value = CumuPixel[nodes[i].value]
    return nodes


def rayleighStrUpper(nodes, height, width,lower_Position):
    allSize = height*width
    alpha = 0.4
    selectedRange = [0, 255]
    NumPixel = np.zeros(256)
    temp = np.zeros(256)
    for i in range(lower_Position, allSize):
            NumPixel[nodes[i].value] = NumPixel[nodes[i].value] + 1
    ProbPixel = NumPixel / (allSize-lower_Position)
    CumuPixel = np.cumsum(ProbPixel)
    valSpread = selectedRange[1] - selectedRange[0]
    hconst = 2 * alpha ** 2
    vmax = 1 - e ** (-1 / hconst)
    val = vmax * (CumuPixel)
    val = np.array(val)

    for i in range(256):
        if (val[i] >= 1):
            val[i] = val[i] - esp
    for i in range(256):
        temp[i] = np.sqrt(-hconst * math.log((1 - val[i]), e))
        normalization = temp[i] * valSpread
        if(normalization > 255):
            CumuPixel[i] = 255
        else:
            CumuPixel[i] = normalization
    for i in range(lower_Position, allSize):
        nodes[i].value = CumuPixel[nodes[i].value]
    return nodes



def uperLower(r, height, width):
    allSize = height * width
    R_max = np.max(r)
    R_min = np.min(r)
    R__middle = (R_max - R_min) / 2 + R_min
    R__middle = np.mean(r)
    node_upper = []
    node_lower = []
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, r[i, j])
            oneNodeLower = NodeLower(i, j, r[i, j])
            node_upper.append(oneNode)
            node_lower.append(oneNodeLower)
    node_upper = sorted(node_upper, key=lambda node: node.value, reverse=False)
    node_lower = sorted(node_lower, key=lambda node: node.value, reverse=False)

    # print('R__middle',R__middle)
    # middle_Position=[]
    for i in range(allSize):
        if (node_upper[i].value > R__middle):
            # print('nodes[i].value',nodes[i].value)
            middle_Position = i
            break
    lower_Position = middle_Position
    # print('lower_Position',lower_Position)

    for i in range(allSize):
        node_upper[i].value = np.int(node_upper[i].value)
        node_lower[i].value = np.int(node_lower[i].value)
    # print('nodes', nodes[0].value)
    # print('nodes[lower_Position + 10].value', nodes[lower_Position + 2].value)

    nodesLower  = rayleighStrLower(node_lower, height, width,lower_Position)
    nodesUpper  = rayleighStrUpper(node_upper, height, width,lower_Position)

    array_lower_histogram_stretching = np.zeros((height, width))
    array_upper_histogram_stretching = np.zeros((height, width))


    for i in range(0, allSize):
        if(i > lower_Position):
            array_upper_histogram_stretching[nodesUpper[i].x, nodesUpper[i].y] = nodesUpper[i].value
            array_lower_histogram_stretching[nodesUpper[i].x, nodesUpper[i].y] = 255
        else:
            array_lower_histogram_stretching[nodesLower[i].x, nodesLower[i].y] = nodesLower[i].value
            array_upper_histogram_stretching[nodesLower[i].x, nodesLower[i].y] = 0

    # print('np.mean(array_lower_histogram_stretching))',np.mean(array_lower_histogram_stretching))
    # print('np.mean(array_upper_histogram_stretching))',np.mean(array_upper_histogram_stretching))

    return array_lower_histogram_stretching,array_upper_histogram_stretching

def rayleighStretching(sceneRadiance, height, width):

    R_array_lower_histogram_stretching, R_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 2], height, width)
    G_array_lower_histogram_stretching, G_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 1], height, width)
    B_array_lower_histogram_stretching, B_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 0], height, width)

    sceneRadiance_Lower = np.zeros((height, width, 3), )
    sceneRadiance_Lower[:, :, 0] = B_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 1] = G_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 2] = R_array_lower_histogram_stretching
    sceneRadiance_Lower = np.uint8(sceneRadiance_Lower)

    sceneRadiance_Upper = np.zeros((height, width, 3))
    sceneRadiance_Upper[:, :, 0] = B_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 1] = G_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 2] = R_array_upper_histogram_stretching
    sceneRadiance_Upper = np.uint8(sceneRadiance_Upper)

    return sceneRadiance_Lower, sceneRadiance_Upper




####################################################################3
def sceneRadianceRGB(sceneRadiance):

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance
#######################################################3

def MSE(img1, img2):
    squared_diff = (img1 -img2)**2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]
    err = summed / num_pix
    
    return err








img = cv2.imread('imm1.jpg')

height = len(img)
width = len(img[0])

sceneRadiance = RGB_equalisation(img, height, width)

sceneRadiance = stretching(sceneRadiance)

sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)
sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

#cv2.imwrite('Lower0.jpg', sceneRadiance_Lower)
#cv2.imwrite('Upper0.jpg', sceneRadiance_Upper)

sceneRadiance = HSVStretching(sceneRadiance)
sceneRadiance = sceneRadianceRGB(sceneRadiance)

value2 = MSE(img, sceneRadiance) 

print(f"Mean square error between images before and after processing: {value2}")

beforered=img[0]
beforegreen=img[1]
beforeblue=img[2]


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









cv2.imwrite('result2.jpg', sceneRadiance)

