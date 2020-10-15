from math import e
import keras
import librosa
from python_speech_features import mfcc
from python_speech_features import delta
import numpy
import os
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle
import argparse
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# base class for our speaker recognition classifier
# it defines the common variables and functions for its two children:
# speaker_recogntion_classifier_trainer and speaker_recogntion_classifier_evaluator
class speaker_recogntion_classifier():
    def __init__(self):
        # downsample the audio signal to 8000HZ to reduce computation complexity
        self.sample_rate = 8000 

        # the threshold (in decibles) below which to consider as silence
        self.silence_cutoff = 30

        # the frame length, in second
        self.winlen = 0.032

        # the FFT(Fast Fourier Transform) size
        self.nfft = 256

        # the number of MFCC coefficients
        self.numcep = 13

        # total number of features
        self.num_features = None

        # keras feedforward nueral network model path
        self.keras_model_path = 'model/speaker_recognition_keras.h5'
        
        # knn model path
        self.knn_model_path = 'model/speaker_recognition_knn.pickle'

        # path the of scaler
        self.scaler_path = 'model/scaler.pickle'

        # path of the label encoder
        self.labelencoder_path = 'model/labelencoder.pickle'


    def extract_feature_for_audio(self, audio_file_path):
        # load the wav file to an array
        signal,sr =librosa.load(audio_file_path, mono=True, sr=self.sample_rate)

        # trim the leading and trailing slience
        signal_trimed, index = librosa.effects.trim(signal, top_db=self.silence_cutoff)

        # extract the mfcc feature, for details about mfcc,
        # see: http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        MFCC = mfcc(
            signal_trimed,  
            self.sample_rate, 
            winlen=self.winlen, 
            winstep = self.winlen/2, 
            winfunc=numpy.hamming, 
            nfft=self.nfft,
            numcep = self.numcep
        )

        # do not use the first mfcc coefficient
        features = MFCC[:, 1:self.numcep]

        # caculate the delta of the mfcc and add to the features
        Delta=delta(MFCC,2)
        features = numpy.column_stack((features,Delta))
            
        # caculate the delta of the delta of the mfcc and add to the features
        Acc=delta(Delta,2)
        features = numpy.column_stack((features,Acc)) 

        # total number of features would be the number of columns of the `features` array
        self.num_features = features.shape[1]

        # each audio file will be transformed to an numpy array with a shape(N, self.num_features)
        # where N is the number of frames that are extracted from the audio file by the mfcc function
        return features

# class for evaluation
class speaker_recogntion_classifier_evaluator(speaker_recogntion_classifier):
    def __init__(self, model_to_use):
        super().__init__()

        # we support a feedforward model and an knn model
        # this variable saves which model we want to use during the evaluation process
        self.model_to_use = model_to_use

        if(model_to_use == 'feedforward'):
            # load the keras model
            try:
                self.model = keras.models.load_model(self.keras_model_path)
            except OSError as error:
                sys.exit("keras feedforward neural network model not found, please tain the keras model first")
        else:
            # load the knn model
            try:
                with open(self.knn_model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except FileNotFoundError as error:
                sys.exit("knn model file not found, please train the knn model first")

        # load the scaler
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # load the label encoder
        with open(self.labelencoder_path, 'rb') as f:
            self.labelencoder = pickle.load(f)
    
    def evaluate(self, audio_file_path):
        # extract feature from the audio file
        # feature would have a shape of (N, self.num_features)
        # where N is the number of frames that are extracted from the audio file by the mfcc function
        try:
            feature = self.extract_feature_for_audio(audio_file_path)
        except FileNotFoundError as error:
            sys.exit('{} could not be found'.format(audio_file_path))

        # scale the feature to zero mean and unit variance
        X = self.scaler.transform(feature)

        # predict the class for each frame
        # the classes are 0, 1, 2, 3, 4 in this example, as we have 5 speakers
        if(self.model_to_use == 'feedforward'):
            yhat = self.model.predict_classes(X)
        else:
            yhat = self.model.predict(X)

        # get the labels back from the class
        # the labels would be speaker2, speaker3, speaker6, speaker8, speaker10 in this example
        labels = self.labelencoder.inverse_transform(yhat)
        print("There are {} frames analysed in total:".format(labels.size))

        # count the number of each labels in the predicted result
        result = Counter(labels).most_common()

        # print the result
        for item in result:
            print("    - {} frames classified as id: {}".format(item[1], item[0]))
        print("The predicted speaker ID is: {}".format(result[0][0]))

# class for training
class speaker_recogntion_classifier_trainer(speaker_recogntion_classifier):
    def __init__(self):
        super().__init__()
        # the feature array for training
        self.X = None

        # the array of class numbers
        # in this example, the class numbers are 0, 1, 2, 3, 4 as we have 5 speakers
        self.y = None

        # in this example, this would be 5
        self.num_categories = None

        # process the training data
        self.data_processing()

    def data_processing(self):
        # specify the directory containing the training audio files
        # each audio file is named as: <speaker_id>_<file_id>.wav
        data_dir = 'training_data/'

        # a variable that will store our features for all the audio files
        X = None
        speaker_ids = []

        # loop through each audio file
        for f in os.listdir(data_dir):
            if(not f.endswith('.wav')):
                # we only process .wav files
                continue

            print("process training data: " + f)

            # each file is named as: <speaker_id>_<file_id>.wav
            # so we extract the speaker id
            speaker_id = f.split('_')[0]

            # extact the features from the audio file
            # the feature would have a shape of (N, self.num_features)
            # where N is the number of frames extracted from the audio file by the mfcc function
            feature = self.extract_feature_for_audio(data_dir + f)

            # we want to make sure we have N speaker_ids corresponding to the N rows of features
            # as the loop goes, the speaker_ids will grow
            speaker_ids += [speaker_id] * feature.shape[0]

            # 
            if(X is None):
                X = feature
            else:
                # as the loop goes, we will append more rows to X
                # NOTE: X always has same number of rows as speaker_ids during and after the loop
                X = numpy.append(X, feature, axis=0)
            
        # scale the features to zero mean and unit variance
        std_scaler=preprocessing.StandardScaler()
        X = std_scaler.fit_transform(X)
        self.X = X

        # save the scaler, so we can use it later in the evaluation phase
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(std_scaler, f)
        
        # transform speaker_ids to class
        # in order for the training algorithm to work, we have to transform string lables
        # (speaker2, speaker3, speaker6, speaker8, speaker10) to class numbers: (0, 1, 2, 3, 4)
        le = preprocessing.LabelEncoder()
        classes = le.fit_transform(speaker_ids)
        self.y = classes
        self.num_categories = numpy.unique(classes).size

        # save the label encoder
        with open(self.labelencoder_path, 'wb') as f:
            pickle.dump(le, f)

    def train_keras(self):
        # class labels to categorical
        # this will transform the class numbers to one-hot encoding
        # in this case, class numbers [0, 1, 2, 3, 4] will be transformed to 
        # [[1,0,0,0,0],
        #  [0,1,0,0,0],
        #  [0,0,1,0,0],
        #  [0,0,0,1,0],
        #  [0,0,0,0,1]]
        y = to_categorical(self.y)

        # split the training data to training portion and testing portion
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size = 0.2, random_state = 2020)

        # build the model
        model = Sequential()
        model.add(Input(shape=(self.num_features,)))
        model.add(Dense(20, activation='sigmoid'))
        # add drop out to prevent overfitting
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='sigmoid'))
        # add drop out to prevent overfitting
        model.add(Dropout(0.2))
        model.add(Dense(self.num_categories, activation = 'softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

        # set up early stopping callback to stop the training process 
        # in this example, stop the training if the validation accuracy does not
        # increase more that 0.5% in 500 epochs
        es = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=500)

        # setup a model checkpoint to save the model during the training proess
        # this is to make sure we save the model that has the max validation accuracy when it stops
        cp = ModelCheckpoint(self.keras_model_path, monitor='val_accuracy', mode='max')

        # train the model
        model.fit(
            X_train, 
            y_train, 
            validation_data = (X_test, y_test), 
            epochs= 5000, 
            batch_size=512,
            verbose=2,
            callbacks=[es, cp]
            )
    
    def train_knn(self):
        # define a knn classifier
        knn = KNeighborsClassifier(n_neighbors=5)

        # split the training data to training portion and testing portion
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=2020)

        # fit the knn classifier
        knn.fit(X_train, y_train)

        # print the accuracy on the testing data set
        print('the validation accuracy is: {}'.format(knn.score(X_test, y_test)))

        # save the model
        with open(self.knn_model_path, 'wb') as f:
            pickle.dump(knn, f)
        
if __name__ == '__main__':

    # command line arguments parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--operation", required=True, help="choose operations", choices=['train', 'evaluate'], default='train')
    ap.add_argument("-m", "--model", required=False, help="which model to be trained/used", choices=['feedforward', 'knn'], default='feedforward')
    ap.add_argument("-f", "--file", required=False, help="file to evaluate")
    args = vars(ap.parse_args())
    
    if(args['operation'] == 'train'):
        # user selected `train`, so we train the model
        trainer = speaker_recogntion_classifier_trainer()
        if(args['model'] == 'feedforward'):
            trainer.train_keras()
        else:
            trainer.train_knn()
    else:
        # user selected `evaluate`, so we use the trained model
        if(args['file'] is None):
            sys.exit('you have to specify an audio file with the -f option')
        # create an evaulator using the model specified from the command line
        evaluator = speaker_recogntion_classifier_evaluator(args['model'])
        # evaluate the audio file to calculate the caller id
        evaluator.evaluate(args['file'])

