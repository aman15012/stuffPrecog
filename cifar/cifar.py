'''
Reference : https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
			https://elitedatascience.com/keras-tutorial-deep-learning-in-python

'''
# Depth of the dataset is 3 for RGB
from keras.datasets import cifar10 #download the cifar 10 dataset and save it in the keras directory
from matplotlib import pyplot as plt
from scipy.misc import toimage # to visualize the data
import numpy as np
from keras.models import Sequential # to build a sequential CNN
from keras.layers import Dense, Dropout, Flatten, Activation # import different types of layers of NN
from keras.constraints import maxnorm
from keras.optimizers import SGD # for gradient descent
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data() # loads the data

plt.imshow(toimage(X_train[0]))
plt.show() # to see the actual image

np.random.seed(0) #fix random seed for reproducable results

# we normalize the data (good practice)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# for a neural net, we need to have labels in the form of a binary matrix

Y_test = np_utils.to_categorical(Y_test)
Y_train = np_utils.to_categorical(Y_train) # works fine because of uniform divison of data in test and train
classes = Y_train.shape[1]

# creating the model

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32),activation='relu')) # (3,3) is kernel size
model.add(Dropout(0.2)) # to prevent overfitting
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # to reduce the size of our matrix by using a max filter
model.add(Flatten()) # flatten before sending to fully connected layer
model.add(Dense(512, activation='relu')) # fully-connected layer
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax')) # output layer of "classes" size

# compilation

epochs = 2 # iterations
lr = 0.01 # learning rate in gd
decay = lr/epochs
sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False) # setting up gradient descent
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# cross-entropy loss, accuracy as metric
print(model.summary())


# Fit 
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32) # we use mini-batch gradient decent with bs = 32

# Accuracy
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]))