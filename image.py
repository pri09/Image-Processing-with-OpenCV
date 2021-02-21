import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
 
###Read the above using the opencv module.
 
image = cv2.imread('C:/Users/hp/Desktop/IM ML OJT/Week 4/image.jpg')
cv2.imshow('Original image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Convert the same into grayscale and display it.

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Display the shape of the image.

print( image.shape )
height, width, channels = image.shape

###Resize the image to (1000x1000) and display it.

imagecopy= np.copy(gray)
resizeimage= cv2.resize(imagecopy, (1000,1000))
cv2.imshow('Resized Image', resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Smoothen an image ( Blur )

blur = cv2.GaussianBlur(resizeimage,(5,5),0)
cv2.imshow('Blurred Image', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Draw on an image. ( a rectangle around any object, blue solid circle in the middle of the image, draw a green line diagonally across the image)
rect = cv2.rectangle(image,(200,20),(126,108),(0,0,255),2)
cir = cv2.circle(rect,(202,98), 40, (255,0,0), -1)
fin = cv2.line(cir,(0,0),(404,196),(0,255,0),2)
cv2.imshow('Drawn on Image', fin)
cv2.waitKey(0)
cv2.destroyAllWindows()

###Perform Edge Detection using Canny Edge Detection

# defining the canny detector function, here weak_th and strong_th are thresholds for double thresholding step 
def Canny_detector(img, weak_th = None, strong_th = None): 
      
    # conversion of image to grayscale 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
       
    # Noise reduction step 
    img = cv2.GaussianBlur(img, (5, 5), 1.4) 
       
    # Calculating the gradients 
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
      
    # Conversion of Cartesian coordinates to polar  
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
       
    # setting the minimum and maximum thresholds for double thresholding 
    mag_max = np.max(mag) 
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image   
    height, width = img.shape 
       
    # Looping through every pixel of the grayscale image 
    for i_x in range(width): 
        for i_y in range(height): 
               
            grad_ang = ang[i_y, i_x] 
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
               
            # selecting the neighbours of the target pixel according to the gradient direction in the x axis direction 
            if grad_ang<= 22.5: 
                neighb_1_x, neighb_1_y = i_x-1, i_y 
                neighb_2_x, neighb_2_y = i_x + 1, i_y 
              
            # top right (diagnol-1) direction 
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction 
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagnol-2) direction 
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle 
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
                neighb_1_x, neighb_1_y = i_x-1, i_y 
                neighb_2_x, neighb_2_y = i_x + 1, i_y 
               
            # Non-maximum suppression step 
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
                    mag[i_y, i_x]= 0
   
    weak_ids = np.zeros_like(img) 
    strong_ids = np.zeros_like(img)               
    ids = np.zeros_like(img) 
       
    # double thresholding step 
    for i_x in range(width): 
        for i_y in range(height): 
              
            grad_mag = mag[i_y, i_x] 
              
            if grad_mag<weak_th: 
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th: 
                ids[i_y, i_x]= 1
            else: 
                ids[i_y, i_x]= 2
       
       
    # finally returning the magnitude of gradients of edges 
    return mag 
   

# calling the designed function for finding edges 
canny_img = Canny_detector(image) 
   
# Displaying the input and output image   
plt.figure() 
f, plots = plt.subplots(2, 1)  
cv2.imshow('Gray image', canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


###Perform Image Thresholding

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 
   
cv2.imshow('Binary Threshold', thresh1) 
cv2.imshow('Binary Threshold Inverted', thresh2) 
cv2.imshow('Truncated Threshold', thresh3) 
cv2.imshow('Set to 0', thresh4) 
cv2.imshow('Set to 0 Inverted', thresh5) 
      
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()


###Detect and draw contours
    
# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 

# Finding Contours 
# Use a copy of the image e.g. edged.copy() since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 

print("Number of Contours found = " + str(len(contours))) 

# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 

cv2.imshow('Contours', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 


  
