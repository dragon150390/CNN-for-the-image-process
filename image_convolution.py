import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join('C:/Users/thinkpad/Documents/','bricks.png')
im_array = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#plt.imshow(im_array)

im = im_array

#let's define the kernel for convolution
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])

print (im.shape)

def convolution(im,kernel):
    result = np.zeros(im.shape)
    # Output array
    for ii in range(im.shape[0] - 3):
        for jj in range(im.shape[1] - 3):
            result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()
    return result



# Print result
plt.imshow(convolution(im,kernel))
plt.show()