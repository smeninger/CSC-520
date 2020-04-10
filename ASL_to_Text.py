import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers

#Prints out the asl_alphabet_test and asl_alphabet_train folders
print(os.listdir('C:/Users/smeni/OneDrive/Desktop/Spring_2020_Classes/Capstone_Part_1/asl-alphabet'))

#Making note of the training and testing directories
train_dir = 'C:/Users/smeni/OneDrive/Desktop/Spring_2020_Classes/Capstone_Part_1/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_dir = 'C:/Users/smeni/OneDrive/Desktop/Spring_2020_Classes/Capstone_Part_1/asl-alphabet/asl_alphabet_test/alphabet_test'

#Prints Unique Labels
def load_unique():
    size_img = 64,64 #Dimensionality
    images_for_plot = [] #All the images 
    labels_for_plot = [] #Labels of the alphabet
    for folder in os.listdir(train_dir): #For the folder located in the training directory
        for file in os.listdir(train_dir + '/' + folder): #For the file(s) located in this directory
            filepath = train_dir + '/' + folder + '/' + file #Filepath
            image = cv2.imread(filepath) #Read the images in the filepath
            final_img = cv2.resize(image, size_img) #Resize the images in the filepath
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB) #Convert the images to this color scale
            images_for_plot.append(final_img) #Passing through the images
            labels_for_plot.append(folder) #Passing through the folders
            break
    return images_for_plot, labels_for_plot 
    
images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot) #Lists out the alphabet letters and the three additional tasks
labels_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P': 15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V':21, 'W':22, 'X':23, 'Y':24, 'Z':25, 'space':26, 'del':27, 'nothing':28}

#Load data
def load_data():
    images = [] #Images to load
    labels = [] #Labels to load
    size = 64,64 #Dimensoinality
    print("LOADING DATA FROM : ", end = '') #Loading data from each of the alphabet/select commands folders
    for folder in os.listdir(train_dir): #For the folder located in the training directory
        print(folder, end = ' | ') #Print all the folders in that directory
        for image in os.listdir(train_dir + "/" + folder): #For the images that are within the specified path
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)#Read the images that are within the individual folders
            temp_img = cv2.resize(temp_img,size)#Resize the images
            images.append(temp_img)#Pass ther images through
            #Pass through all these images
            if folder == 'A':
                labels.append(labels_dict['A'])
            elif folder == 'B':
                labels.append(labels_dict['B'])
            elif folder == 'C':
                labels.append(labels_dict['C'])
            elif folder == 'D':
                labels.append(labels_dict['D'])
            elif folder == 'E':
                labels.append(labels_dict['E'])
            elif folder == 'F':
                labels.append(labels_dict['F'])
            elif folder == 'G':
                labels.append(labels_dict['G'])
            elif folder == 'H':
                labels.append(labels_dict['H'])
            elif folder == 'I':
                labels.append(labels_dict['I'])
            elif folder == 'J':
                labels.append(labels_dict['J'])
            elif folder == 'K':
                labels.append(labels_dict['K'])
            elif folder == 'L':
                labels.append(labels_dict['L'])
            elif folder == 'M':
                labels.append(labels_dict['M'])
            elif folder == 'N':
                labels.append(labels_dict['N'])
            elif folder == 'O':
                labels.append(labels_dict['O'])
            elif folder == 'P':
                labels.append(labels_dict['P'])
            elif folder == 'Q':
                labels.append(labels_dict['Q'])
            elif folder == 'R':
                labels.append(labels_dict['R'])
            elif folder == 'S':
                labels.append(labels_dict['S'])
            elif folder == 'T':
                labels.append(labels_dict['T'])
            elif folder == 'U':
                labels.append(labels_dict['U'])
            elif folder == 'V':
                labels.append(labels_dict['V'])
            elif folder == 'W':
               labels.append(labels_dict['W'])
            elif folder == 'X':
                labels.append(labels_dict['X'])
            elif folder == 'Y':
                labels.append(labels_dict['Y'])
            elif folder == 'Z':
                labels.append(labels_dict['Z'])
            elif folder == 'space':
                labels.append(labels_dict['space'])
            elif folder == 'del':
                labels.append(labels_dict['del'])
            elif folder == 'nothing':
                labels.append(labels_dict['nothing'])

        images = np.array(images) #Array values for the images
        images = images.astype('float32')/255.0

        labels = keras.utils.to_categorical(labels)

        X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.1) #Size of the data being tested

        print()
        print('Loaded', len(X_train), 'images for training,','Train data shape = ',X_train.shape)
        print('Loaded', len(X_test), 'images for testing','Test data shape =',X_test.shape)

        return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_data()


#Build Model
def build_model():
    model = Sequential() #takes an array of Keras Layers

    model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(MaxPool2D(3))

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    #Dense Layers
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(29, activation = 'softmax'))

    #Compile the model
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

    print("MODEL CREATED")
    model.summary()
    return model

#Fit Model
def fit_model():
    history = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split = 0.1)
    return history
model = build_model()
model_history = fit_model()
        

                     



            
               

    

