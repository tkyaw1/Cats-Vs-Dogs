import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from numpy import argmax, zeros, logical_not

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, Flatten

# TODO: load test file (if you want)
# NOTE: relu for non output layers, sigmoid for output layers ?? not sure y
# NOTE: adding another pair of pooling/conv layer dropped accuracy a lot

def settingItUp():
    neural_net = Sequential()

    neural_net.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', input_shape = (768, 1050, 3)))
    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Conv2D(128, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Conv2D(64, (3, 3), activation = 'relu'))
    neural_net.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    neural_net.add(Flatten())
    neural_net.add(Dense(32, activation = 'relu'))
    neural_net.add(Dropout(0.5))
    neural_net.add(Dense(2, activation = 'sigmoid'))
    #neural_net.add(Dropout(0.2))
    neural_net.summary()

    neural_net.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    return neural_net

def train(xtrainfinal, ytrainfinal, neural_net):
    history = neural_net.fit(xtrainfinal, ytrainfinal, verbose=1, epochs=10)

def test(xtest, ytest, neural_net):
    """Reports the fraction of the test set that is correctly classified.
    The predict() keras method is run on each element of x_test, and the result
    is compared to the corresponding element of y_test. The fraction that
    are classified correctly is tracked and returned as a float.
    """
    loss, accuracy = neural_net.evaluate(xtest, ytest, verbose=0)
    return accuracy

def crossValidation():
    neural_net = settingItUp()
    folds = 10
    files = []
    labels = []
    for j in range(10):
        files.append("/scratch/tkyaw1/outfile" + str(j) + ".npz")
        labels.append("/scratch/tkyaw1/labels" + str(j) + ".npz")
    files = np.array(files)
    labels = np.array(labels)

    # filesSmallSubset = "/scratch/tkyaw1/smallSubset.npz"
    # labelsSmallSubset = "/scratch/tkyaw1/smallLabels.npz"

    percentlist = []
    for i in range(folds):
        print "FOLD NUMBER:", i
        b = zeros(10, dtype = bool)
        bcopy = b
        start = i*1
        end = (i+1) * 1
        bcopy[start:end] = True

        xtrain = files[logical_not(bcopy)]
        trainLabels = labels[logical_not(bcopy)]
        xtest = files[bcopy]
        testLabels = labels[bcopy]
        print "xtrain:", xtrain
        print "trainLabels:", trainLabels
        print "xtest:", xtest
        print "testLabels:", testLabels

        for j in range(len(xtrain)):
            outfile = np.load(xtrain[j])
            loadedOutfile = outfile['arr_0']

            trainingLabels = np.load(trainLabels[j])
            loadedLabels = trainingLabels['arr_0']

            print "TRAINING XTRAIN [j]:", xtrain[j]
            train(loadedOutfile, loadedLabels, neural_net)

        foldAccs = []
        for x in range(len(xtest)):
            outfile = np.load(xtest[x])
            loadedOutfile = outfile['arr_0']

            testingLabels = np.load(testLabels[x])
            loadedLabels = testingLabels['arr_0']

            print "TESTING XTEST [x]:", xtest[x]
            foldAcc = test(loadedOutfile, loadedLabels, neural_net)
            print "foldAcc:", foldAcc
            foldAccs.append(foldAcc)

        accuracy = sum(foldAccs)/float(len(foldAccs))
        print "Accuracy over entire fold", accuracy
        #testing
        percentlist.append(accuracy)

    average = sum(percentlist)/float(len(percentlist))
    print "Final Average Accuracy Over 10 Folds", average

crossValidation()
