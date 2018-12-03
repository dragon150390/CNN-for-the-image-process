import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import xlrd

import utils

DATA_FILE = 'data/fire_theft.xls'

# read data form excel file 

book = xlrd.open_workbook(DATA_FILE,encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(i,sheet.nrows)])
n_data = sheet.nrows -1 

# create placeholders for input X , Y
Y = tf.placeholders(tf.float32,name='Y')
X = tf.placeholders(tf.float32,name='X')

# create W and bias with initialize value is zero
w = tf.Variable(0.0,name='weights')
bias = tf.variable(0.0,name='bias')

#build the model in tf
Y_predicted = w*X + bias

#definec the roor squared error for loss function
loss = tf.square(Y-Y_predicted,name='loss')
#Using gradient descent with learning rate euqal 0.01 to minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
def huber_loss_function(Y, predicted,delta=14.):
    residual = tf.abs(Y-predicted)
    f1 = lambda: 0.5*tf.square(residual)
    f2 = lambda: delta*residual - 0.5*tf.square(delta)
    result = tf.cond(residual <= delta,f1,f2)
    return result

with tf.Session() as sess:
    #initialize the variables w and bias 

    sess.run(tf.global_variables_initializer())

    #start the task save the graph for tensorflow


    writer = tf.summary.FileWriter('./graphs/linear_reg',sess.graph)

    #train the model
    for i in range(50) #we will train the data with 50 epochs
        for x , y in data:
            _,l = sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss += l
        print ('Epoch {0} : {1}'.format(i,total_loss/n_data))
    writer.close()
    w,b = sess.run([w,b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()