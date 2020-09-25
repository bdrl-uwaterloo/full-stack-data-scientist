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
        self.img_width = 160
        self.img_height = 160
        self.model_save_path = 'model/model_cnn.h5'
        self.class_labels = ['NORMAL', 'PNEUMONIA']

# this is the class that will use the trained model to evaluate new x-ray images
class chest_xray_classifier_evaluator(chest_xray_classifier):
    def __init__(self):
        super().__init__()
        if(os.path.exists(self.model_save_path)):
            self.model = load_model(self.model_save_path)
            print(self.model.summary())
        else:
            self.model = None

    def evaluate(self, image_path):
        if(self.model):
ayscale", target_size=(self.img_height,             img = load_img(image_path, color_mode = "grself.img_width))
            img_data = img_to_array(img)  
            img_data = np.array([img_data])
            result = self.model.predict_classes(img_data) #shape of result is (1, 1)
            return self.class_labels[result[0][0]]
        else:
            return 'Could Not Find Model'
                
# this is the class that we will use to train the model
class chest_xray_classifier_trainer(chest_xray_classifier):
    def __init__(self):
        super().__init__()
        self.train_data_dir = 'data/train'
        self.validation_data_dir = 'data/test'

        self.nb_train_samples = self.get_number_of_files(self.train_data_dir)
        self.nb_validation_samples = self.get_number_of_files(self.validation_data_dir)
        self.epochs = 30
        self.batch_size = 32

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
        model.add(Conv2D(32, (2, 2), input_shape = self.input_shape, kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        model.add(Conv2D(32, (2, 2), kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        model.add(Conv2D(64, (2, 2), kernel_regularizer=l2(0.001), activation='relu')) 
        model.add(MaxPooling2D(pool_size =(2, 2)))
        
        model.add(Flatten()) 
        model.add(Dense(50, activation='relu')) 
        model.add(Dropout(0.5)) 
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
            rescale = 1. / 255, 
            shear_range = 0.2,
            zoom_range = 0.2,
            rotation_range= 15,
            width_shift_range= 0.1,         
            height_shift_range= 0.1,         
            horizontal_flip = False) 
        
        # we don't have to augment test data, just need to scale the data
        test_datagen = ImageDataGenerator(rescale = 1. / 255) 

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir, 
            target_size =(self.img_height, self.img_width), 
            color_mode='grayscale',
            batch_size = self.batch_size, 
            class_mode ='binary') 

        print(train_generator.class_indices)
        
        validation_generator = test_datagen.flow_from_directory( 
            self.validation_data_dir, 
            target_size =(self.img_height, self.img_width), 
            color_mode='grayscale',
            batch_size = self.batch_size, 
            class_mode ='binary')

        model.fit(
            train_generator, 
            steps_per_epoch = math.ceil(self.nb_train_samples / self.batch_size), 
            epochs = self.epochs, 
            validation_data = validation_generator, 
            validation_steps = math.ceil(self.nb_validation_samples / self.batch_size))
        
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

    option = 'cnn'
    if(len(sys.argv) == 2):
        option = sys.argv[1]

    if(option == 'cnn'):
        cxc.train_cnn()
    elif(option == 'feedforward'):
        cxc.train_feedforward()