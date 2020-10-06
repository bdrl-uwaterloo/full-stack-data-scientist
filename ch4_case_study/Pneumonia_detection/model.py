from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.regularizers import l2
from keras import backend as K 
import tensorflow as tf
import os
import sys
import math
import numpy as np

# base class for our chest x-ray classifier
class chest_xray_classifier():
    def __init__(self):
        # define the width and height for the images that will be fed into the CNN.
        # images in the training set will be re-scalled to have the following width and height.
        self.img_width = 160
        self.img_height = 160

        # save location of the model
        self.model_save_path = 'model/model_cnn.h5'

        # class labels 0: 'NORMAL', 1: 'PNEUMONIA'
        # 
        self.class_labels = ['NORMAL', 'PNEUMONIA']

# this is the class that will use the trained model to evaluate new x-ray images
class chest_xray_classifier_evaluator(chest_xray_classifier):
    def __init__(self):
        # call base class __init__ function to initialize base class' attributes
        super().__init__()

        # if we have a trained model already, we load it here
        if(os.path.exists(self.model_save_path)):
            self.model = load_model(self.model_save_path)
            print(self.model.summary())
        else:
            self.model = None

    def evaluate(self, image_path):
        if(self.model):
            # we load the image as grayscale to memory and scale it
            img = load_img(image_path, color_mode = "grayscale", target_size=(self.img_height, self.img_width))

            # convert image data to a numpy array
            img_data = img_to_array(img) # this has a shape of (height, width, channels)

            # add an extra dimension to make sure it can be fed into the CNN, as Keras NN expects an array of input:
            # (number_of_images, height, width, channels)
            img_data = np.array([img_data])
            result = self.model.predict_classes(img_data) #shape of result is (1, 1)
            return self.class_labels[result[0][0]]
        else:
            return 'Could Not Find Model'
                
# this is the class that we will use to train the model
class chest_xray_classifier_trainer(chest_xray_classifier):
    def __init__(self):
        super().__init__()
        # training data and verification data directory
        self.train_data_dir = 'data/train'
        self.validation_data_dir = 'data/test'

        self.nb_train_samples = self.get_number_of_files(self.train_data_dir)
        self.nb_validation_samples = self.get_number_of_files(self.validation_data_dir)
        self.epochs = 30
        self.batch_size = 32

        # image data includes channel, width and height. different Keras backend uses different format 
        # make sure we use the correct format for the corresponding backend.
        # in this example, we use Tensorflow as the backend.
        if K.image_data_format() == 'channels_first': 
            self.input_shape = (1, self.img_height, self.img_width) 
        else: 
            self.input_shape = (self.img_height, self.img_width, 1)

    # return number of files in directory folder and its sub-directories
    def get_number_of_files(self, folder):
        nb_files = 0
        for root, dirs, files in os.walk(folder):
            nb_files += len(files)
        return nb_files

    # build a CNN model, which has 3 (convolutional layers + max pooling layers) as feature extractors
    # and a fully connected layer in then end as the classifier
    # TODO: add a picture to illustrate the network structure 
    def train_cnn(self):
        model = Sequential() 

        # a first Conv2D + MaxPooling2D layer as feature extractor
        # it has 32 filters of size 2X2
        # TODO: briefly explain the purpose of regulization is to prvent overfit
        model.add(Conv2D(32, (2, 2), input_shape = self.input_shape, kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        # a second Conv2D + MaxPooling2D layer as feature extractor
        model.add(Conv2D(32, (2, 2), kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        # a third Conv2D + MaxPooling2D layer as feature extractor
        model.add(Conv2D(64, (2, 2), kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2)))
        
        # flattern the output of previous layer so we can feed it to a fully connected layer
        model.add(Flatten()) 

        # add a Dense(or fully connected layer) as a classifier
        model.add(Dense(50, activation='relu')) 

        # add a Dropout layer to prvent overfit
        model.add(Dropout(0.5)) 

        # this is the output layer
        model.add(Dense(1, activation='sigmoid')) 
        
        # compile the model structure, specify loss and optimizer and metrics
        model.compile(
            loss ='binary_crossentropy', 
            optimizer ='adam', 
            metrics =['accuracy']) 

        # print the model structure 
        model.summary()

        # use ImageDataGenerator to prepare our training data
        # ImageDataGenerator is a great tool to augment our training data
        # [TODO] add image examples on what this tool can do, such as zoom, shear, rotate, shift. 
        train_datagen = ImageDataGenerator( 
            # image data has values in the range of [0, 255]. rescale the image data to 0-1
            rescale = 1. / 255, 
            
            shear_range = 0.2,
             
            zoom_range = 0.2, 
            
            rotation_range= 15,
            
            width_shift_range= 0.1,
            
            height_shift_range= 0.1,
            
            horizontal_flip = False) 
        
        # we don't have to augment test data, just need to scale the data
        test_datagen = ImageDataGenerator(rescale = 1. / 255) 
        
        # prepare the training data. this function automatically discover the class names based on the folder names
        # and sort them by alphabetical order. So under data/train/, we have two folders, NORMAL and PNEUMONIA, this function 
        # will use 0 for NORMAL and 1 for PNEUMONIA
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir, 
            # rescale the image
            target_size =(self.img_height, self.img_width), 
            # we only need grayscale for this task
            color_mode='grayscale',
            batch_size = self.batch_size, 
            # we only have two classes: NORMAL and PNEUMONIA
            class_mode ='binary') 

        print(train_generator.class_indices)
        
        validation_generator = test_datagen.flow_from_directory( 
            self.validation_data_dir, 
            target_size =(self.img_height, self.img_width), 
            color_mode='grayscale',
            batch_size = self.batch_size, 
            class_mode ='binary')

        # train the model
        model.fit(
            train_generator, 
            steps_per_epoch = math.ceil(self.nb_train_samples / self.batch_size), 
            epochs = self.epochs, 
            validation_data = validation_generator, 
            validation_steps = math.ceil(self.nb_validation_samples / self.batch_size))
        
        # save the model
        model.save(self.model_save_path) 

    # to compare, we also build a basic feed forward network with similary amount of total parameters (you can check the model.summary() output)
    # and show the performance of this model
    def train_feedforward(self):
        model = Sequential() 
        
        model.add(Flatten(input_shape = self.input_shape)) 
        model.add(Dense(46, activation='relu')) 
        model.add(Dropout(0.5)) 
        model.add(Dense(35, activation='relu')) 
        model.add(Dropout(0.5)) 
        model.add(Dense(1, activation='sigmoid')) 
        
        model.compile(
            loss ='binary_crossentropy', 
            optimizer ='adam', 
            metrics =['accuracy']) 

        model.summary()
        
        train_datagen = ImageDataGenerator( 
            rescale = 1. / 255, 
            shear_range = 0.2, 
            zoom_range = 0.2,  
            width_shift_range= 0.1,
            height_shift_range= 0.1,
            horizontal_flip = False) 
        
        test_datagen = ImageDataGenerator(rescale = 1. / 255) 
        
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir, 
            target_size =(self.img_width, self.img_height), 
            color_mode='grayscale',
            batch_size = self.batch_size, 
            class_mode ='binary') 

        
        validation_generator = test_datagen.flow_from_directory( 
            self.validation_data_dir, 
            target_size =(self.img_width, self.img_height), 
            color_mode='grayscale',
            batch_size = self.batch_size, 
            class_mode ='binary') 
        
        model.fit(
            train_generator, 
            steps_per_epoch = self.nb_train_samples // self.batch_size + 1, 
            epochs = self.epochs, 
            validation_data = validation_generator, 
            validation_steps = self.nb_validation_samples // self.batch_size + 1) 

if __name__ == '__main__':
    cxc = chest_xray_classifier_trainer()

    #default option is to train the cnn model
    option = 'cnn'
    if(len(sys.argv) == 2):
        option = sys.argv[1]

    if(option == 'cnn'):
        # train a model using cnn
        cxc.train_cnn()
    elif(option == 'feedforward'):
        # train a model using feed forward network
        cxc.train_feedforward()