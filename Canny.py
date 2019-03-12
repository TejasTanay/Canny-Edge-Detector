import cv2
import numpy as np 
import math


def main():
	imgg = cv2.imread('Lena256.bmp',cv2.IMREAD_GRAYSCALE)  
	#Type in the full path of the image or iff the image is in the same directory as the code then type only the name of the image
	#this function will store the gray intensity values in a 2D array called imgg
	size=imgg.shape;
	#This stores the row and column of the input image in variables m and n (in lines below)
	n=imgg.shape[0];
	m=imgg.shape[1];

	mG,nG,imggGaus=Gaussian(m,n,imgg)
	#Gaussian Function does the Gaussian smoothing, it takes three arguents the 1)row and 2)column value of the pixels and the 
	#3)image 2D matrix 
	# and it returns three values the smoothened image in an 1)2D array, and also the 2)row and 3)column value of this aray
	
	mP,nP,Px,Py=Prewitt(mG,nG,imggGaus)
	#Prewitt Function does the Gradient operation, it takes three arguments the 1)row and 2)column value of the pixels and the 
	#3)Gaussian smoothened image 2D matrix 
	#and it returns four values the smoothened image in an 1)2D array of Horizontal gradient, 2) 2D array of vertical gradient 
	#and also the 2)row and 3)column value of these aray 
	
	mNMS,nNMS,NMS=NMS1(mP,nP,Px,Py)
	#NMS1 Function does the Non Maxima Supression, it takes four arguments the 1)2D array of Horizontal gradient, 2) 2D array of vertical gradient 
	#and also the 2)row and 3)column value of these aray
	# and it returns three values the magnitude array after non maxima suppression in an 1)2D array, and also the 2)row and 3)column value of this aray


	Ptile(mNMS,nNMS,NMS)
	#Ptile Function does thresholding, it takes three arguments the 1)row and 2)column value of the pixels and the 
	#3)Non maxima suppressed Magnitude image 2D matrix
	# It does not return any value but prints 3 images with 10%,30% and 50% thresholding
	






def Gaussian(m,n,imgg):
	mG=m-6# column value of the gaussian filtered image which will be produced
	nG=n-6# row value of the gaussian filtered image which will be produced
	imggGaus=np.zeros((nG,mG)) #Intializing all the values in the gaussian filtered image (called imggGaus) to 0
	
	#Convolution happening below with 7X7 Gaussian Mask
	for i in range(3,n-3):
		for j in range(3,m-3):
			imggGaus[i-3][j-3]=((imgg[i-3][j-3]*1)+(imgg[i-3][j-2]*1)+(imgg[i-3][j-1]*2)+(imgg[i-3][j]*2)+(imgg[i-3][j+1]*2)+(imgg[i-3][j+2]*1)+(imgg[i-3][j+3]*1)+
				(imgg[i-2][j-3]*1)+(imgg[i-2][j-2]*2)+(imgg[i-2][j-1]*2)+(imgg[i-2][j]*4)+(imgg[i-2][j+1]*2)+(imgg[i-2][j+2]*2)+(imgg[i-2][j+3]*1)+
				(imgg[i-1][j-3]*2)+(imgg[i-1][j-2]*2)+(imgg[i-1][j-1]*4)+(imgg[i-1][j]*8)+(imgg[i-1][j+1]*4)+(imgg[i-1][j+2]*2)+(imgg[i-1][j+3]*2)+
				( imgg[i][j-3]*2)+  (imgg[i][j-2]*4)+  (imgg[i][j-1]*8)+  (imgg[i][j]*16)+ (imgg[i][j+1]*8)+  (imgg[i][j+2]*4)+  (imgg[i][j+3]*2)+
				(imgg[i+1][j-3]*2)+(imgg[i+1][j-2]*2)+(imgg[i+1][j-1]*4)+(imgg[i+1][j]*8)+(imgg[i+1][j+1]*4)+(imgg[i+1][j+2]*2)+(imgg[i+1][j+3]*2)+
				(imgg[i+2][j-3]*1)+(imgg[i+2][j-2]*2)+(imgg[i+2][j-1]*2)+(imgg[i+2][j]*4)+(imgg[i+2][j+1]*2)+(imgg[i+2][j+2]*2)+(imgg[i+2][j+3]*1)+
				(imgg[i+3][j-3]*1)+(imgg[i+3][j-2]*1)+(imgg[i+3][j-1]*2)+(imgg[i+3][j]*2)+(imgg[i+3][j+1]*2)+(imgg[i+3][j+2]*1)+(imgg[i+3][j+3]*1))/140

	cv2.imwrite("GaussianOf.bmp",imgg)
	return mG,nG,imggGaus

def Prewitt(mG,nG,imggGaus):
	mP=mG-2# column value of the Gradient images which will be produced
	nP=nG-2# row value of the Gradient images which will be produced
	
	#Gx

	Px=np.zeros((nP,mP))#Intializing all the values in the Gx image (called Px) to 0
	#Convolution happening below with Prewitt's Horizontal Gradient 3X3 mask
	for i in range(1,nG-1):
		for j in range(1,mG-1):
			Px[i-1][j-1]=((imggGaus[i-1][j-1]*-1)+(imggGaus[i-1][j]*0)+(imggGaus[i-1][j+1]*1)+(imggGaus[i][j-1]*-1)+(imggGaus[i][j]*0)+(imggGaus[i][j+1]*1)+(imggGaus[i+1][j-1]*-1)+(imggGaus[i+1][j]*0)+(imggGaus[i+1][j+1]*1))
			if (Px[i-1][j-1])<0:
				Px[i-1][j-1]=(Px[i-1][j-1]*-1)
			Px[i-1][j-1]=Px[i-1][j-1]/3
	cv2.imwrite("Px.bmp",Px)
	
	#Gy

	Py=np.zeros((nP,mP))#Intializing all the values in the Gy image (called Py) to 0
		#Convolution happening below with Prewitt's Vertital Gradient 3X3 mask
	for i in range(1,nG-1):
		for j in range(1,mG-1):
			Py[i-1][j-1]=((imggGaus[i-1][j-1]*1)+(imggGaus[i-1][j]*1)+(imggGaus[i-1][j+1]*1)+(imggGaus[i][j-1]*0)+(imggGaus[i][j]*0)+(imggGaus[i][j+1]*0)+(imggGaus[i+1][j-1]*-1)+(imggGaus[i+1][j]*-1)+(imggGaus[i+1][j+1]*-1))
			if (Py[i-1][j-1])<0:
				Py[i-1][j-1]=(Py[i-1][j-1]*-1)
			Py[i-1][j-1]=Py[i-1][j-1]/3

	cv2.imwrite("Py.bmp",Py)
	return mP,nP,Px,Py

def NMS1(mP,nP,Px,Py):
	#Calculating Magnitude with the help of Gx and Gy
	Magnitude=np.zeros((nP,mP))
	for i in range(nP):
		for j in range(mP):
			Magnitude[i][j]=math.sqrt((Px[i][j]*Px[i][j])+(Py[i][j]*Py[i][j]))
			Magnitude[i][j]=Magnitude[i][j]/np.sqrt(2)

	cv2.imwrite("Magnitude.bmp",Magnitude)

	mNMS=mP-2 # column value of the Non Maxima Supressed Magnitude image which will be produced
	nNMS=nP-2 # row value of the Non Maxima Supressed Magnitude image which will be produced

	NMS=np.zeros((nNMS,mNMS)) #Intializing all the values in the NMS image (called NMS) to 0

	#Calculating the Gradient angle, then identifying which sector it belongs to and then doing the magnitude comparisons with its neighbours(sector defined)
	for i in range(1,nP-1):
		for j in range(1,mP-1):
			if (Px[i][j]!=0):
				deg= (math.degrees(math.atan(Py[i][j]/Px[i][j])))
			else:
				if(Py[i][j]>0):
					deg=90
				else:
					deg=-90

			if (deg<0):
				deg=deg+360

			if (337.5<= deg <360) or (0<= deg <22.5) or (157.5<= deg <202.5):
				if (Magnitude[i][j]>Magnitude[i][j-1]) and (Magnitude[i][j]>Magnitude[i][j+1]):
					NMS[i-1][j-1]=Magnitude[i][j]
				else:
					NMS[i-1][j-1]=0
			if (22.5<= deg <67.5) or (202.5<= deg <247.5):
				if (Magnitude[i][j]>Magnitude[i-1][j+1]) and (Magnitude[i][j]>Magnitude[i+1][j-1]):
					NMS[i-1][j-1]=Magnitude[i][j]
				else:
					NMS[i-1][j-1]=0

			if (67.5<= deg <112.5) or (247.5<= deg <292.5):
				if (Magnitude[i][j]>Magnitude[i-1][j]) and (Magnitude[i][j]>Magnitude[i+1][j]):
					NMS[i-1][j-1]=Magnitude[i][j]
				else:
					NMS[i-1][j-1]=0
			if (112.5<= deg <157.5) or (292.5<= deg <337.5):
				if (Magnitude[i][j]>Magnitude[i-1][j-1]) and (Magnitude[i][j]>Magnitude[i+1][j+1]):
					NMS[i-1][j-1]=Magnitude[i][j]
				else:
					NMS[i-1][j-1]=0


	cv2.imwrite("NMS.bmp",NMS)
	return mNMS,nNMS,NMS

def Ptile(mNMS,nNMS,NMS):
	NMS10=np.zeros((nNMS,mNMS))#Intializing all the values in the threshold 10  image (called NMS10) to 0
	NMS30=np.zeros((nNMS,mNMS))#Intializing all the values in the threshold 30  image (called NMS30) to 0
	NMS50=np.zeros((nNMS,mNMS))#Intializing all the values in the threshold 50  image (called NMS50) to 0
	pixxy =[] #this is an array which stores all edge point's grey intensity value after Non Maxima suppression
	counter=0 #stores the number of edge points after thresholding
	for i in range(nNMS):
 		for j in range(mNMS):
 			if (NMS[i][j]>0):
 				pixxy.append(NMS[i][j])


	pixxy.sort(reverse=True) # all the grey intensity value of the edge points are sorted in reverse order
	total=len(pixxy) #number of total edge points
	print ("total edge after NMS pixels")
	print (total)
	edges=total

	count=0
	total=total*0.1
	total=int(total)
	total=total-1 # this is done to get the correct index value in the pixxy array
	threshold=round(pixxy[total]) #threshold at 10
	print ("threshold 10")
	print(threshold)
	for i in range(nNMS):
 		for j in range(mNMS):
 			if(NMS[i][j]<threshold):
 				NMS10[i][j]=0
 			else:
 				NMS10[i][j]=255
 				count=count+1
	
	print("edge pixes at threshold 10%")
	print(count)
	cv2.imwrite("NMS_10.bmp",NMS10)

	count=0
	total=edges
	total=total*0.3
	total=int(total)
	total=total-1
	threshold=round(pixxy[total])
	print ("threshold 30")
	print(threshold)
	for i in range(nNMS):
 		for j in range(mNMS):
 			if(NMS[i][j]<threshold):
 				NMS30[i][j]=0
 			else:
 				NMS30[i][j]=255
 				count=count+1
	print("edge pixes at threshold 30%")
	print(count)
	cv2.imwrite("NMS_30.bmp",NMS30)

	count=0
	total=edges
	total=total*0.5
	total=int(total)
	total=total-1
	threshold=round(pixxy[total])
	print ("threshold 50")
	print(threshold)
	for i in range(nNMS):
 		for j in range(mNMS):
 			if(NMS[i][j]<threshold):
 				NMS50[i][j]=0
 			else:
 				NMS50[i][j]=255
 				count=count+1
	print("edge pixes at threshold 50%")
	print(count)
	cv2.imwrite("NMS_50.bmp",NMS50)

if __name__ == "__main__":
	main()

