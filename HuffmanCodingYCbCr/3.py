import numpy as np 
from PIL import Image 
import math

#def plot(lis):
#	print(lis)
#	search={}

#	for index in range(len(lis)):
#		name="block"+str(index+1)
#		search[lis[index]]=name
	
#	layer=0
#	layers=[]
	
#	for i in range(len(lis)):
#		if len(lis[i])>layer:
#			layer=len(lis[i])

#	for _ in range(layer):
#		layers.append([])

#	for k in range(len(lis)):
#		layers[len(lis[k])-1].append(lis[k])

#	print(layers)
#	print(search)
#	firstspace=15
#	print("               start") #15 space
#	for n in range(len(layers)):
#		firstspace-=1
#		for s in range(firstspace):
#			print(" ", end='')
#		print("node", end='')
#		for _ in range(n+1):
#			print(" ", end='')
#		print("node")
		



def stat(arr):
	batch1Count=0
	batch2Count=0
	batch3Count=0
	batch4Count=0
	batch5Count=0
	totalpixel=arr.shape[0]*arr.shape[1]

	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
			if arr[i][j]<=51:
				batch1Count+=1
			if (arr[i][j]>51) and (arr[i][j]<=102):
				batch2Count+=1
			if (arr[i][j]>102) and (arr[i][j]<=153):
				batch3Count+=1
			if (arr[i][j]>153) and (arr[i][j]<=204):
				batch4Count+=1
			if (arr[i][j]>204) and (arr[i][j]<=255):
				batch5Count+=1
	batch1prob=batch1Count/totalpixel
	batch2prob=batch2Count/totalpixel
	batch3prob=batch3Count/totalpixel
	batch4prob=batch4Count/totalpixel
	batch5prob=batch5Count/totalpixel

	namee="block     probability"
	print(namee)
	print("block0    "+str(batch1prob))
	print("block1    "+str(batch2prob))
	print("block2    "+str(batch3prob))
	print("block3    "+str(batch4prob))
	print("block4    "+str(batch5prob))

	#
	#test=[batch1prob,batch2prob,batch3prob,batch4prob,batch5prob]
	#print(test)
	#

	wait={'1':batch1prob,'2':batch2prob,'3':batch3prob,'4':batch4prob,'5':batch5prob}
	sortwait={k: v for k, v in sorted(wait.items(), key=lambda item: item[1])}

	encodename=[]
	for _ in range(len(wait.keys())):
		encodename.append([])         #reverse

	while len(sortwait)>1:
		temp=list(sortwait)
		smallest=temp[0]
		secondsmall=temp[1]

		for m in range(len(smallest)):
			encodename[int(smallest[m])-1].append(0)

		for n in range(len(secondsmall)):
			encodename[int(secondsmall[n])-1].append(1)

		comwho=smallest+secondsmall
		comprob=sortwait[smallest]+sortwait[secondsmall]
		wait.pop(smallest)
		wait.pop(secondsmall)

		wait[comwho]=comprob
		sortwait={k: v for k, v in sorted(wait.items(), key=lambda item: item[1])}

	finalans=[]
	for u in range(len(encodename)):
		tarr=encodename[u]
		last=len(tarr)-1
		ans=''
		for k in range(last,-1,-1):
			ans+=str(tarr[k])

		finalans.append(ans)

	search={}

	for index in range(len(finalans)):
		name="block"+str(index)
		search[name]=finalans[index]

	return search


def cal(imagee):
	img = Image.open(imagee)
	img2=img.convert(mode="RGB")
	r,g,b=img2.split()

	rr=np.array(r)
	gg=np.array(g)
	bb=np.array(b)

	color=["red","green","blue"]
	colorarr=[rr,gg,bb]

	title="encoding result for "+imagee+":"
	print(title)
	for g in range(len(color)):
		sub=color[g]+" channel:"
		print(sub)
		print(stat(colorarr[g]))
	
	print('\n')	

def bitstreamtest(ima):
	img = Image.open(ima)
	img2=img.convert(mode="RGB")
	r,g,b=img2.split()

	print("test for bitstream of red channel:")
	rr=np.array(r)
	se=stat(rr)

	for i in range(rr.shape[0]):
		for j in range(rr.shape[1]):
			bitstreamm=0
			if rr[i][j]<=51:
				bitstreamm=se['block0']
			if (rr[i][j]>51) and (rr[i][j]<=102):
				bitstreamm=se['block1']
			if (rr[i][j]>102) and (rr[i][j]<=153):
				bitstreamm=se['block2']
			if (rr[i][j]>153) and (rr[i][j]<=204):
				bitstreamm=se['block3']
			if (rr[i][j]>204) and (rr[i][j]<=255):
				bitstreamm=se['block4']

			print(bitstreamm, end='')

	




cal('foreman_qcif_0_rgb.bmp')
cal('foreman_qcif_1_rgb.bmp')
cal('foreman_qcif_2_rgb.bmp')

bitstreamtest('foreman_qcif_0_rgb.bmp')