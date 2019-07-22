import numpy as np
import cv2

I = cv2.imread('/home/alireza/Documents/cv-lab4/branches2.jpg').astype(np.float64) / 255

noise_sigma = 0.04  # initial standard deviation of noise

m = 1  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size
sigmaColor = 0.3
sigmaSpace = 75

# with m = 1 the input image will not change
filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    J = (I + N).astype(np.float32)

    if filter == 'b':
        # filter with a box filter
        F = np.ones( (m, m), np.float64 ) / (m * m)
        K = cv2.filter2D( J, -1, F )
    elif filter == 'g':
        # filter with a Gaussian filter
        Fg = cv2.getGaussianKernel( m, sigma=-1 )
        F = Fg.dot( Fg.T )
        K = cv2.filter2D( J, -1, F )
    elif filter == 'l':
        # filter with a bilateral filter
        K = cv2.bilateralFilter( J, size, sigmaColor, sigmaSpace )

    # filtered image

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print('Box filter')

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print('Gaussian filter')

    elif key == ord('l'):
        filter = 'l'  # filter with a bilateral filter
        print('Bilateral filter')

    elif key == ord('+'):
        # increase m
        m = m + 2
        print('m=', m)

    elif key == ord('-'):
        # decrease m
        if m >= 3:
            m = m - 2
        print('m=', m)
    elif key == ord('u'):
        # increase noise
        noise_sigma += 0.1
    elif key == ord('d'):
        # decrease noise
        noise_sigma -= 0.1
    elif key == ord('p'):
        # increase gm
        sigmaColor += 0.3
    elif key == ord('n'):
        # decrease gm
        sigmaColor -= 0.3
    elif key == ord('>'):
        # increase size
        size +=4
        print("size ",size)
    elif key == ord('<'):
        # decrease size
        size -= 4
        print( "size ", size )
    elif key == ord('q'):
        break  # quit

cv2.destroyAllWindows()
