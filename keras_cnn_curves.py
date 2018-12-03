import matplotlib.pyplot as plt 

#train model and store the training object

traning = model.fit(train_data,train_labels,epochs=3,validation_split=0.2)

#extract the history from traning object

history = traning.history


# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()
