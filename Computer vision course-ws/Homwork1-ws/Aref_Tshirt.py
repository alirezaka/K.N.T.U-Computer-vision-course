import cv2
import numpy as np

cap=cv2.VideoCapture()

cap.open('/home/alireza/Documents/Homework 1/Make him invisible.avi')

stackedFrame = []
stackedCFrame = []

minHSV = np.array( [100, 10, 50] )
maxHSV = np.array( [155, 180, 210] )
kernel = np.ones( (5, 5), np.uint8 )

w=640
h=480
fourcc = cv2.VideoWriter_fourcc(*'XVID') # choose codec
out = cv2.VideoWriter('Aref_Tshirt.avi',fourcc, 30.0, (w,h))

while True:


     ret, I = cap.read()

     if ret == False:
          exit(0)
     else:

          hsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
          mskHSV = cv2.inRange( hsv, minHSV, maxHSV )

          cv2.medianBlur( mskHSV, 11 )
          T = cv2.erode(mskHSV,kernel)
          T= cv2.dilate(T,kernel)

          stackedFrame.append(T)
          stackedCFrame.append(I)

          J= np.zeros(T.shape)
          W = np.zeros( T.shape )

          med=stackedFrame[int(len(stackedFrame)/2)]
          medColor=stackedCFrame[int(len(stackedCFrame)/2)]

          G=np.abs(T-med)
          J[G > 250]=255

          result=I.copy()

          cv2.medianBlur( np.uint8(J), 7 )


          msk1= cv2.bitwise_and(medColor, medColor, mask = np.uint8(J) )

          W[J==0] = 255
          W[J==255] = 0

          msk2 = cv2.bitwise_and( I, I, mask=np.uint8( W ) )
          cv2.add(msk1,msk2,result)

          cv2.imshow("result",result)
          out.write(result)
          q = cv2.waitKey( 33 )
          if q == ord( 'q' ):
               exit( 0 )
