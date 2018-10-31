#Import matplotlib
import matplotlib.pyplot as plt

# Load data from image
data = plt.imread('c:/Users/thinkpad/Documents/bricks.png')
# -- modify the bricks image to replace the top left corner of the image (10 X 10)  in to red square --
# Set the red chanel in the part of image to 1
data[:10,:10,0] = 1
# Set the green chanel in the part of image to 0
data[:10,:10,1] = 0
# Set the blue chanel in the part of image to 0
data[:10,:10,2] = 0
#Display Image
plt.imshow(data)
plt.show()