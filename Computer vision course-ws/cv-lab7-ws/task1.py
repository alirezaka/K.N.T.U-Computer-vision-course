import numpy as np
import cv2

I = cv2.imread( '/home/alireza/Documents/cv-lab7/coins.jpg' )
G = cv2.cvtColor( I, cv2.COLOR_BGR2GRAY )
G = cv2.GaussianBlur( G, (5, 5), 0 );

canny_high_threshold = 160
min_votes = 30 # minimum no. of votes to be considered as a circle
min_centre_distance = 40

circles = cv2.HoughCircles(G,cv2.HOUGH_GRADIENT,1,min_centre_distance,param1=canny_high_threshold,param2=min_votes,minRadius=0,maxRadius=100)


for c in circles[0, :]:
    x = c[0]
    y = c[1]
    r = c[2]
    cv2.circle( I, (x, y), r, (0, 255, 0), 2 )



n = circles[0,:].shape[0]
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText( I, 'There are %d coins!' % n, (400, 40), font, 1, (255, 0, 0), 2 )

cv2.imshow( "I", I )
cv2.waitKey( 0 )

