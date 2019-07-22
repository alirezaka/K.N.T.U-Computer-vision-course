import cv2
import numpy as np

cap=cv2.VideoCapture()

cap.open('/home/alireza/Documents/Homework 1/Make him invisible.avi')

stackedFrame = []
stackedCFrame = []

hist = np.zeros( (1, 16), dtype=np.float32 )

hist[0, 0] = 255
hist[0, 1] = 255
hist[0, 15] = 255
hist[0, 8] = 255
hist[0, 9] = 255
hist[0, 10] = 255

minHSV_blue = np.array( [90, 0, 0] )
maxHSV_blue = np.array( [140, 255, 255] )

minHSV_red = np.array( [0, 60, 50] )
maxHSV_red = np.array( [20, 170, 170] )

kernel = np.ones((3,3),np.uint8)

while True:

     ret, I = cap.read()

     if ret == False:
          exit(0)
     else:

          hsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)

          hue=np.zeros(hsv.shape,dtype=np.uint8)


          cv2.mixChannels([hsv],[hue],fromTo=[0,0])


          bp=cv2.calcBackProject([hue],[1,0],hist,[0,180,0,256],1)


          mskHSV_blue=cv2.inRange(hsv,minHSV_blue,maxHSV_blue)

          mskHSV_red=cv2.inRange(hsv,minHSV_red,maxHSV_red)

          mskHSV = np.zeros( (480,640),dtype=np.uint8 )


          cv2.addWeighted( mskHSV_red, 1, mskHSV_blue, 1, 0, mskHSV)

          bp = mskHSV & bp

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

          J = cv2.erode( J, kernel )
          J = cv2.dilate( J, kernel )

          msk1= cv2.bitwise_and(medColor, medColor, mask = np.uint8(J) )

          W[J==0] = 255
          W[J==255] = 0

          msk2 = cv2.bitwise_and( I, I, mask=np.uint8( W ) )
          cv2.add(msk1,msk2,result)

          cv2.imshow("result",result)
          q = cv2.waitKey( 33 )
          if q == ord( 'q' ):
               exit( 0 )
