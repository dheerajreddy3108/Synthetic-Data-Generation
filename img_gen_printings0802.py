# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:21:12 2021

@author: studperadh6230
"""

import numpy as np
import cv2
import os
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from random import randint


#defect free image to get defects placed at required location
# we make copies so that we don#t effect original image

bg_name = 'D:/manual_defectfree/outputs_0802/output_def_free_D/imgs-'+str(1)+'.png'
bg_img = Image.open(bg_name,mode='r')

ref_img1 = bg_img.copy()
ref_img2 = bg_img.copy()

ref_img1 = np.asarray(ref_img1)
ref_img2 = np.asarray(ref_img2)

#selecting defect number using user input
def_location = 'D:/manual_defectfree/defects/*.png'
total_defects = len(glob.glob(def_location))
defects = []

for i in range(total_defects):
    img_name = 'D:/manual_defectfree/defects/'+str(i+1) +'.png'
    img = Image.open(img_name,mode='r')
    img = img.resize((30,30),resample = PIL.Image.LANCZOS)
    defects.append(img)
    fig = plt.figure()
    plt.axis('off') 
    a = fig.add_subplot(1, 2, i+1)
    a.set_title(str(i))
    imgplot = plt.imshow(img)
    

def_num = int(input('choose a defect type number : 0 or 1: ' ))
temp_img = defects[def_num]
plt.imshow(temp_img, cmap = 'gray')

#mouse operation to select a color using mouse
# we use mouse-left button for these operation

def mouseRGB(event,x,y,flags,param):
    global red, green, blue
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        red = np.int(colorsR)
        green = np.int(colorsG)
        blue = np.int(colorsB)
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BGR Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
    
    return red,green,blue


# Read an image, a window and bind the function to window
ref_img2 = cv2.imread('D:/manual_defectfree/outputs_0802/output_def_free_D/imgs-'+str(1)+'.png')
cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

#Do until esc pressed
while(1):
    cv2.imshow('mouseRGB',ref_img2)
    if cv2.waitKey(20) & 0xFF == 27:
        break
#if esc pressed, finish.
cv2.destroyAllWindows()


im = Image.open('D:/petraGlass4DF_8.png',mode='r')
R, G, B = im.convert('RGB').split()
r = R.load()
g = G.load()
b = B.load()

#we ask user to specify location for missing pixels 
missing_point = []
def missing_px_point(event,x,y,flags,params):
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(ref_img2,(x,y),radius =1,color = (255,255,255))
        missing_point.append((x,y))
    
    return missing_point

cv2.namedWindow('missing_pixel_trail')
cv2.setMouseCallback('missing_pixel_trail',missing_px_point)
  
while True:
	# display the image and wait for a keypress
	cv2.imshow("missing_pixel_trail", ref_img2)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, copy reference image
	if key == ord("r"):
            image = ref_img2.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
            break
    
cv2.destroyAllWindows()

ref_img1 =bg_img.copy()
R, G, B = ref_img1.convert('RGB').split()
r = R.load()
g = G.load()
b = B.load()

w = 10
h = 10

# Convert non-black pixels to white
if (missing_point[0][0]-w & missing_point[0][0]+w  & missing_point[0][1]-h & missing_point[0][1]+h) !=0:
    
    for i in range(missing_point[0][0]-w,missing_point[0][0]+w):
        for j in range(missing_point[0][1]-h,missing_point[0][1]+h):
            if(r[i, j] != red or g[i, j] != green or b[i, j] != blue):
                r[i,j] = red
                g[i,j] = green
                b[i,j] = blue
            
# Merge all channels
ref_img1 = Image.merge('RGB', (R, G, B))
ref_img1.save("D:/black_and_white.png")

plt.imshow(ref_img1)


#checking for point inside polygon

INT_MAX = 10000

# Given three colinear points p, q, r, 
# the function checks if point q lies 
# on line segment 'pr' 
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
	
	if ((q[0] <= max(p[0], r[0])) &
		(q[0] >= min(p[0], r[0])) &
		(q[1] <= max(p[1], r[1])) &
		(q[1] >= min(p[1], r[1]))):
		return True
		
	return False



# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p:tuple, q:tuple, r:tuple) -> int:
	
	val = (((q[1] - p[1]) *
			(r[0] - q[0])) -
		((q[0] - p[0]) *
			(r[1] - q[1])))
			
	if val == 0:
		return 0 # Collinear
	if val > 0:
		return 1 # Clockwise
	else:
		return 2 # Counterclockwise

a = (10,10)
c = (100,10)
b = (20,20)

#v = orientation(a,b,c)
#print(v)

def doIntersect(p1, q1, p2, q2):
	
	# Find the four orientations needed for 
	# general and special cases 
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if (o1 != o2) and (o3 != o4):
		return True
	
	# Special Cases 
	# p1, q1 and p2 are colinear and 
	# p2 lies on segment p1q1 
	if (o1 == 0) and (onSegment(p1, p2, q1)):
		return True

	# p1, q1 and p2 are colinear and 
	# q2 lies on segment p1q1 
	if (o2 == 0) and (onSegment(p1, q2, q1)):
		return True

	# p2, q2 and p1 are colinear and 
	# p1 lies on segment p2q2 
	if (o3 == 0) and (onSegment(p2, p1, q2)):
		return True

	# p2, q2 and q1 are colinear and 
	# q1 lies on segment p2q2 
	if (o4 == 0) and (onSegment(p2, q1, q2)):
		return True

	return False

# Returns true if the point p lies 
# inside the polygon[] with n vertices 
def is_inside_polygon(points:list, p:tuple) -> bool:
	
	n = len(points)
	
	# There must be at least 3 vertices
	# in polygon
	if n < 3:
		return False
		
	# Create a point for line segment
	# from p to infinite
	extreme = (INT_MAX, p[1])
	count  = i = 0
    
    
	
	while True:
		next = (i + 1) % n
		
		# Check if the line segment from 'p' to 
		# 'extreme' intersects with the line 
		# segment from 'polygon[i]' to 'polygon[next]' 
		if (doIntersect(points[i],
						points[next], 
						p, extreme)):
							
			# If the point 'p' is colinear with line 
			# segment 'i-next', then check if it lies 
			# on segment. If it lies, return true, otherwise false 
			if orientation(points[i], p, 
						points[next]) == 0:
				return onSegment(points[i], p, 
								points[next])
								
			count += 1
			
		i = next
		
		if (i == 0):
			break
		
	# Return true if count is odd, false otherwise 
	return (count % 2 == 1)



polygon1 = [(0,0),(10,0),(10,10),(0,10)]
     
p = (5,5)
#checking with images
img = np.zeros((256,256),dtpye = np.uint8)


if (is_inside_polygon(points = polygon1, p = p)): 

       cv2.line(img,(10,10),(100,100),color=(127,127,127),thickness=1)
  
plt.imshow(img)


def_num = int(input('choose a defect type number : 0 or 1: ' ))
temp_img = defects[def_num]
plt.imshow(temp_img, cmap = 'gray')
# mouse event to record reference points for the ROI selection 

ref_point = []

def roi_point_click(event,x,y,flags,params):
    
    if event==cv2.EVENT_RBUTTONDBLCLK:
        # create a circle at that position
        # of radius 2 and color greeen
        cv2.circle(ref_img1,(x,y),1,(0,255,0),-1)
        ref_point.append((x,y))
        
    return ref_point

cv2.namedWindow('trail')        
cv2.setMouseCallback('trail',roi_point_click)  

   
while True:
	# display the image and wait for a keypress
	cv2.imshow("trail", ref_img1)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, copy reference image
	if key == ord("r"):
            image = ref_img1.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
            break
    
cv2.destroyAllWindows()

    


polygon1 = ref_point
#getting random point coordinates
     
rand_x = randint(0, 512)
rand_y = randint(0,512)
rand_pt = (rand_x,rand_y)

render_img = bg_img.copy()
#checking with images
if (is_inside_polygon(points = polygon1, p = rand_pt)): 
       print('yes')
       n_count = 1
       for n in range(n_count):
           
           cnt = 1
           for c in range(cnt):
               a = randint(0,360)
               img_size = 30
               w = int(img_size*1)
        
               img2 =  temp_img
               img2 = img2.resize((30,30),resample = PIL.Image.LANCZOS)
               x = rand_x
               y = rand_y 
        #x = randint(-int(img2.width/2),background.width-int(img2.width/2))
        #y = randint(-int(img2.height/2),background.height/2-int(img2.height/2))
        
               img2 = PIL.Image.Image.rotate(img2,a,resample=PIL.Image.BICUBIC,expand=True)
        
               render_img.paste(img2,(x,y),img2)
    
       img_file = BytesIO()
       render_img.save('D:/manual_defectfree/img-01.png')
    

#multi-roi for multiple defects


m = 100

bg_name = 'D:/manual_defectfree/outputs_0802/output_def_free_D/imgs-'+str(m)+'.png'
bg_img = Image.open(bg_name,mode='r')

ref_img1 = bg_img.copy()
ref_img2 = bg_img.copy()

ref_img1 = np.asarray(ref_img1)
ref_img2 = np.asarray(ref_img2)

render_img = bg_img.copy()


def_count = int(input('Please specify number of defects to be added : '))

for i in range(def_count):
    
    def_num = int(input('choose a defect type number : 0 or 1: ' ))
    temp_img = defects[def_num]
    plt.imshow(temp_img, cmap = 'gray')
# mouse event to record reference points for the ROI selection 

    ref_point = []

    def roi_point_click(event,x,y,flags,params):
    
        if event==cv2.EVENT_RBUTTONDBLCLK:
        # create a circle at that position
        # of radius 2 and color greeen
            cv2.circle(ref_img1,(x,y),1,(0,255,0),-1)
            ref_point.append((x,y))
        
        return ref_point

    cv2.namedWindow('trail')        
    cv2.setMouseCallback('trail',roi_point_click)  
    while(True):
        cv2.imshow('trail',ref_img1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = ref_img1.copy()
            
        elif key == ord("c"):
            break
    cv2.destroyAllWindows()
    
#drawing a polygon

    for i in range(len(np.asarray(ref_point)+1)):
        if (i +1) %len(np.asarray(ref_point)) == 0:
            cv2.line(ref_img1,ref_point[i],ref_point[0],color = (255,255,255))
        else:
            cv2.line(ref_img1, ref_point[i],ref_point[i+1],color = (255,255,255))
        plt.imshow(ref_img1,cmap='gray')

    polygon1 = ref_point
#getting random point coordinates
     
    rand_x = randint(0, 255)
    rand_y = randint(0,255)
    rand_pt = (rand_x,rand_y)
#checking with images
    if (is_inside_polygon(points = polygon1, p = rand_pt)): 
        print('yes')
            
        cnt = 1
        for c in range(cnt):
                a = randint(0,360)
                img_size = 30
                w = int(img_size*1)
        
                img2 =  temp_img
                img2 = img2.resize((30,30),resample = PIL.Image.LANCZOS)
                x = rand_x
                y = rand_y 
       
                img2 = PIL.Image.Image.rotate(img2,a,resample=PIL.Image.BICUBIC,expand=True)
        
                render_img.paste(img2,(x,y),img2)
    
        img_file = BytesIO()
        render_img.save('D:/manual_defectfree/img-'+str(m)+'.png')
        plt.imshow(render_img)
    




name = 'D:/manual_defectfree/def0802/img-'+str(91) + '.png'

img = cv2.imread(name)
save_name = 'D:/manual_defectfree/def_bmp/img-'+str(91)+'.bmp'
cv2.imwrite(save_name, img)


