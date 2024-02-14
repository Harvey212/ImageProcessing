import numpy as np 
import cv2
from PIL import Image
from matplotlib import pyplot as plt 
import math

def sob(imm):
    row=imm.shape[0]
    col=imm.shape[1]


    newim=np.zeros((row,col))

    windowsizee=3

    threshold=60

    for i in range(0,row):
        for j in range(0,col):
            centerrow=i
            centercol=j
            span=int((windowsizee-1)/2)
        
            up=centerrow-span
            down=centerrow+span
            left=centercol-span
            right=centercol+span

            localleft=0
            localright=2

            localup=0
            localdown=2

            if up<0:
                up=0
                localup=1
            if down>(row-1):
                down=row-1
                localdown=1
            if left<0:
                left=0
                localleft=1
            if right>(col-1):
                right=col-1
                localright=1

            test=imm[up:(down+1),left:(right+1)]

            SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            kernelx=SobelX[localup:(localdown+1),localleft:(localright+1)]
            kernely=SobelY[localup:(localdown+1),localleft:(localright+1)]

            resx=np.sum(np.multiply(test,kernelx))
            resy=np.sum(np.multiply(test,kernely))

            check=math.sqrt((resx**2+resy**2))

            if check>threshold:
                newim[i][j]=255
            else:
                newim[i][j]=0

    return newim


def marr(imm2):
    row2=imm2.shape[0]
    col2=imm2.shape[1]


    newim2=np.zeros((row2,col2))

    windowsizee2=5

    threshold2=100
    
    for i in range(0,row2):
        for j in range(0,col2):
            centerrow2=i
            centercol2=j
            span2=int((windowsizee2-1)/2)
        
            localleft2=0
            localright2=4

            localup2=0
            localdown2=4

            up2=centerrow2-span2
            down2=centerrow2+span2
            left2=centercol2-span2
            right2=centercol2+span2

            if up2<0:
                if up2==-2:
                    localup2=2
                else:
                    localup2=1
                up2=0
            if down2>(row2-1):
                if (down2-(row2-1))==2:
                    localdown2=2
                else:
                    localdown2=3
                down2=row2-1
            if left2<0:
                if left2==-2:
                    localleft2=2
                else:
                    localleft2=1
                left2=0
            if right2>(col2-1):
                if (right2-(col2-1))==2:
                    localright2=2
                else:
                    localright2=3
                right2=col2-1

            filt=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
            
            
            test2=imm2[up2:(down2+1),left2:(right2+1)]

            kernel=filt[localup2:(localdown2+1),localleft2:(localright2+1)]
            check2=np.sum(np.multiply(test2,kernel))
                    
            if check2>threshold2:
                newim2[i][j]=255
            else:
                newim2[i][j]=0
            

    return newim2




   



tt = cv2.imread("peppers.bmp")
img = cv2.cvtColor(tt, cv2.COLOR_BGR2GRAY)

f1=sob(img)

cv2.imshow('Sobel operator',f1)

cv2.waitKey(0)
cv2.destroyAllWindows()

#########################################

f2=marr(img)
cv2.imshow('Marr-Hildreth operator',f2)

cv2.waitKey(0)
cv2.destroyAllWindows()

