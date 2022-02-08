########Image Processing With Neural Networks
##Images as data: visualizations
# Import matplotlib
import matplotlib.pyplot as plt
# Load the image
data = plt.imread('bricks.png')
# Display the image
plt.imshow(data)
plt.show()

##Images as data: changing images
#Modify the bricks image to replace the top left corner of the image 
#(10 by 10 pixels) into a red square.
# Set the red channel in this part of the image to 1
data[:10,:10,0] = 1
# Set the green channel in this part of the image to 0
data[:10,:10,1] = 0
# Set the blue channel in this part of the image to 0
data[:10,:10,2] = 0
# Visualize the result
plt.imshow(data)
plt.show()

###Classifying images
##Using one-hot encoding to represent images
# The number of image categories
n_categories = 3
# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])
# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))
# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii,jj] = 1

#use above array to test classification performance
# Calculate the number of correct predictions
number_correct = (test_labels*predictions).sum()
print(number_correct)
# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(test_labels)
print(proportion_correct)

###fitting classification models using Keras
##Build a neural network
# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense
# Initializes a sequential model
model = Sequential()
# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))
# Second layer
model.add(Dense(10, activation='relu'))
# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
           loss='categorical_crossentropy', 
           metrics=['accuracy'])
			   
#Fitting a neural network model to clothing data
# Reshape the data to two-dimensional array
train_data = train_data.reshape(50, 784)
# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)
#epochs=3 --go through whole dataset 3 times

#Cross-validation for neural network evaluation
# Reshape test data
test_data = test_data.reshape(10, 784)
# Evaluate the model
model.evaluate(test_data, test_labels)
#The model cross-validates rather accurately.

##############Using Convolutions
##One dimensional convolutions
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)
#Notice that we've only multiplied the kernel with eight different positions

##Image convolutions
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
print(result)
#In a future exercise, you will see how to use Keras to implement a convolution like this one.

##define the kernel that finds a particular feature in the image
#the following kernel finds a vertical line in images:
np.array([[-1, 1, -1], 
          [-1, 1, -1], 
          [-1, 1, -1]])
#kernel that finds horizontal lines in images.
kernel = np.array([[-1, -1, -1], 
                   [1, 1, 1],
                   [-1, -1, -1]])
#kernel that finds a light spot surrounded by dark pixels
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])
#A light spot has a bright pixel (with larger values, e.g., 1) in the center, surrounded by pixels that are dark (lower values, e.g., -1)
#Define a kernel that finds a dark spot surrounded by bright pixels.
kernel = np.array([[1, 1, 1], 
	               [1, -1, 1],
                   [1, 1, 1]])

###Implementing image convolutions in Keras
##Convolutional network for image classification
# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# Initialize the model object
model = Sequential()
# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
               input_shape=(img_rows,img_cols,1)))

# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(3, activation='softmax'))

# Compile the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)
			  
# Evaluate the model on separate test data
model.evaluate(test_data,test_labels,batch_size=10)

###Tweaking your convolutions
##Add padding to a CNN
# Initialize the model
model = Sequential()
# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='same'))
# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
#padding set to 'same', the output layer will have the same size as the input layer

##Add strides to a convolutional network
# Initialize the model
model = Sequential()
# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1), 
              strides=2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
#With strides set to 2, the network skips every other pixel

#######Going deeper
##Creating a deep learning network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='same'))
# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

##Train a deep CNN to classify clothing images
# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)
# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)

###How many parameters?
# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
# Summarize the model 
model.summary()

##Pooling operations
# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2,jj*2:jj*2+2])
#The resulting image is smaller, but retains the salient features in every location

##Keras pooling layers
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
# Add a pooling operation
model.add(MaxPool2D(2))
# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()
#This model is even deeper, but has fewer parameters.

##Train a deep CNN with pooling to classify images
# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, epochs=3,batch_size=10,validation_split=0.2)

# Evaluate on test data 
model.evaluate(test_data,test_labels,batch_size=10)

#######Understanding and Improving Deep Convolutional Networks
###Tracking learning
#Plot the learning curves
import matplotlib.pyplot as plt
# Train the model and store the training object
training = model.fit(train_data,train_labels, validation_split=0.2,batch_size=10,epochs=3)
# Extract the history from the training object
history = training.history
# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])
# Show the figure
plt.show()

from keras.callbacks import ModelCheckpoint
# This checkpoint object will store the model parameters
# in the file "weights.hdf5"
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss',save_best_only=True)
# Store in a list to be used during training
callbacks_list = [checkpoint]
# Fit the model on a training set, using the checkpoint as a
#callback
model.fit(train_data, train_labels, validation_split=0.2,epochs=3, callbacks=callbacks_list)

#Using stored weights to predict in a test set
# Load the weights from file
model.load_weights('weights.hdf5')
# Predict from the first three images in the test data
model.predict(test_data[:3])

####Regularization
##Adding dropout to your network
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
# Add a dropout layer
model.add(Dropout(0.2))
# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

##Add batch normalization to your network
# Add a convolutional layer
model.add(Conv2D(15,kernel_size=2,activation='relu',input_shape=(img_rows, img_cols, 1)))
# Add batch normalization layer
model.add(BatchNormalization())
# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#####Interpreting the model
# Load the weights into the model
model.load_weights('weights.hdf5')
# Get the first convolutional layer from the model
c1 = model.layers[0]
# Get the weights of the first convolutional layer
weights1 = c1.get_weights()
# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][...,0, 0] #=[:, :,0, 0]
print(kernel)

#Visualizing kernel responses
import matplotlib.pyplot as plt
# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)
# Visualize the result
plt.imshow(out)
plt.show()

def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0]-2):
        for jj in range(image.shape[1]-2):
            result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)

    return result