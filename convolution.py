# multiply the input array with value of kernel and sum up multiplied  result and end up to correction location in the output result
# in one the dimemsion
import numpy as np 
# Let's define some array that we will use in this sample
input_array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([-1,1])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# output array 

for ii in range(8):
    conv[ii] = (kernel*input_array[ii:ii+2]).sum()

print (conv)