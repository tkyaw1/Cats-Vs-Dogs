"""
Preprocessing file. Reads in all training images and creates 3 dimensional
arrays from them, putting them in 20 different files each with 1000 image arrays
of dimension (768, 1050, 3). Also creates a label file for all 20000 images.
"""

import numpy as np
from os import listdir as ls
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from random import shuffle


#ndimage.imread("filename")
#plt.imshow(a)
def train_label_img(img):
    word_label = img.split('.')[-3] #get dog/cat out of label
    if word_label == "/scratch/tkyaw1/train/dog":
        return [0,1]
    elif word_label == "/scratch/tkyaw1/train/cat":
        return [1,0]

def main():
    pictures = ls("/scratch/tkyaw1/train/")
    shuffle(pictures)
    maxWidth = 0
    maxHeight = 0
    for p in pictures:
        filename = "/scratch/tkyaw1/train/" + p
        im = Image.open(filename)
        width = im.size[0] # im.size returns (width, height) tuple
        height = im.size[1]
        if width>maxWidth:
            maxWidth = width
        if height>maxHeight:
            maxHeight = height


    for chunk in range(39):
        ytrain = []
        xtrain = []
        start = 0
        end = 500
        for pic in pictures[start:end]:
            trainPic = []
            trainPic = np.zeros([maxHeight, maxWidth, 3])
            filename = "/scratch/tkyaw1/train/" + pic
            a = ndimage.imread(filename)
            im = Image.open(filename)
            pWidth = im.size[0]
            pHeight = im.size[1]
            trainPic[0:pHeight,0:pWidth, :] = a
            trainPic = trainPic / float(255)
            label = train_label_img(filename)
            ytrain.append(label)
            xtrain.append(trainPic)
            start += 500
            end += 500
        np.savez_compressed('/scratch/tkyaw1/outfile'+ str(chunk) + '.npz', xtrain)
        np.savez_compressed('/scratch/tkyaw1/labels' + str(chunk) + '.npz', ytrain)

    """
    ytrain = []
    xtrain = []
    start = 19500
    end = 20024
    for pic in pictures[start:end]:
        trainPic = []
        trainPic = np.zeros([maxHeight, maxWidth, 3])
        filename = "/scratch/tkyaw1/train/" + pic
        a = ndimage.imread(filename)
        im = Image.open(filename)
        pWidth = im.size[0]
        pHeight = im.size[1]
        trainPic[0:pHeight,0:pWidth, :] = a
        trainPic = trainPic / float(255)
        label = train_label_img(filename)
        ytrain.append(label)
        xtrain.append(trainPic)

    np.savez_compressed('/scratch/tkyaw1/outfile'+ str(39) + '.npz', xtrain)
    np.savez_compressed('/scratch/tkyaw1/labels' + str(39) + '.npz', ytrain)
    """
    ytrain = []
    start = 0
    end = 1000
    for pic in pictures[start:end]:
         trainPic = []
         filename = "/scratch/tkyaw1/train/" + pic
         label = train_label_img(filename)
         ytrain.append(label)
    np.savez_compressed('/scratch/tkyaw1/labels' + str(0)+ '.npz', ytrain)


    ytrain = []
    for pic in pictures:
         filename = "/scratch/tkyaw1/train/" + pic
         label = train_label_img(filename)
         ytrain.append(label)
    np.savez_compressed('/scratch/tkyaw1/labels.npz', ytrain)


        # for pic in xtrain:


        # plt.imshow(a)
        # plt.show()

        # pictures = ls("/scratch/tkyaw1/test/")
        # xtest = []
        # for pic in pictures:
        #     testPic = []
        #     filename = "/scratch/tkyaw1/test/" + pic
        #     a = ndimage.imread(filename)
        #     testPic.append(a)
        #     xtest.append(testPic)

        # xtrain = np.array(xtrain)
        # print a
        # ytrain = np.array(ytrain)

        # print trainPic
        # plt.imshow(a)
        # plt.show()
        #
        # plt.imshow(trainPic)
        # plt.show()


    np.savez_compressed('/scratch/tkyaw1/outfile.npz', xtrain)
    a = np.load('outfile.npz')



main()
