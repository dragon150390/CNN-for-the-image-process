# Onece We need to import the keras callback
from keras.callbacks import ModelCheckpoint

#The checkpoint object will store the model parameters

checkpoint = ModelCheckpoint('weights.hdf5',monitor='val_loss', save_best_only=True)
#Store in the list during the training

callback_list = [checkpoint]

# Fit the model in the trainning set using the model callback function

model.fit(train_data,train_labels, validation_split=0.2,epochs=3,callbacks=callback_list)