from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout

model = Sequential()


#Add a convolutional layer
model.add(Conv2D(15,kernel_size=2,activation='relu',input_shape=(img_rows,img_cols,1)))
#Add a drop layer
model.add(Dropout(0.25))

model.add(Conv2D(15,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))