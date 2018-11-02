import numpy as np 


def get_data():
    data = np.load("c:/Users/thinkpad/Documents/fashion.npz")
    print (type(data))
    a = isinstance(data, np.lib.npyio.NpzFile)
    new_data = data['arr_0']
    new_data = new_data.reshape((1,))
    #print (new_data[0]['train_data'])
    train_data = new_data[0]['train_data']
    test_data = new_data[0]['test_data']
    test_labels = new_data[0]['test_labels']
    return train_data , test_data , test_labels

