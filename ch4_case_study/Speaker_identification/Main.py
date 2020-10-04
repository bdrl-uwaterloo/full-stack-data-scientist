import os
import pandas
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank
import numpy
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct
from scipy.signal import argrelextrema
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from itertools import groupby
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#############################################################################################  
#1. extrac_features -----------------------------------------------------------------------------------------------------
def extract_mfcc(pathwav, outputpath):
    files = os.listdir(pathwav)   
    for f in files:
        try:
            if f.endswith(".rar") : continue
            print("process %s" %(f) )
            #if not f.endswith(".wav") or f.startswith('xinglin'): continue
            signal,sr =librosa.load(pathwav+f, mono=True,sr=sample_rate)
            signal_trimed, index = librosa.effects.trim(signal, top_db=silence_cutoff)
            print("total lenth %.2f, after trimming: %.2f " % (len(signal)/sample_rate, len(signal_trimed)/sample_rate ))

            # make sure signal has a length that contains whole frames, cut residuals
            signal_len = signal_trimed.size
            new_len = int(signal_len/frame_stride) * frame_stride
            signal_trimed = signal_trimed[:new_len]
            # Compute the features
            librosa.feature.mfcc(y=signal_trimmed, sr=sr, n_mfcc=13)
            #-----------MFCC-----------------
            MFCC = mfcc(signal_trimed,  sample_rate, winlen=frame_size_seconds, winstep = frame_stride_seconds, winfunc=numpy.hamming, nfft=nfft, numcep = num_cep)
            # do not use the first MFCC coef
            MFCC = MFCC[:, 1:num_cep]

            # make sure MFCC and pitches have the same number of frames 
            # as they come from different library which does different things in the end of the singal
            names = ['MFCC']*num_cep_useful
            features= MFCC  
            
            #-----------Delta-----------------#
            Delta=delta(MFCC,2)
            names = names + ['delta']*num_cep_useful
            features = numpy.column_stack((features,Delta))
                
            #----------acceleration------------------#
            Acc=delta(Delta,2)
            names = names + ['Acc']*num_cep_useful
            features = numpy.column_stack((features,Acc)) 

            # #-----------partials------------#
            # partials = get_harmonics(signal_trimed, pitches)
            # names = names + ['partials']*10
            # features = numpy.column_stack((features, partials))

            #std_scale=preprocessing.StandardScaler().fit(features)
            #features_nm=std_scale.transform(features)
            df = pandas.DataFrame(features,columns=names)
            df.to_csv(outputpath+f+"_features.csv", index = False, header=True, sep=',')
        except:
            print(f) 
    print("all features extracted")
        
#2. merge  ----------------------------------------------------------------------------------------------------- 
def merged_train(outputpath,pathsave):
    filelist = os.listdir(outputpath)
    merged = pandas.DataFrame([])
    Error_ID=[]
    for f in filelist:  
        try:
            if f.endswith("train_merged.csv"): continue
            print('================processing'+f+' ==================')
            readf=pandas.read_csv(outputpath+f)
            cnames =list(range(1,readf.shape[1]+1))
            readf.columns=cnames
            f_id=f.split('_',-1)[0]
            readf['id']=[f_id]*readf.shape[0]
            merged=merged.append(readf,sort=False,ignore_index=True) 
        except:
            Error_ID.append(f)
    print(Error_ID)

    features=merged.loc[:, merged.columns != 'id'].values

    label=pandas.factorize(merged.id)[0]

    merged.to_csv(outputpath+"train_merged.csv", index = False, header=True)
    
    print("================ pretrained_merged ==================")
    return (features,label)


# train model (keras) -----------------------------------------------------------------------------------------------------
def class_model(total_num_features,hidden_dim,encode_dim,epochs,pathsave,features,label):
    std_scaler=preprocessing.StandardScaler().fit(features)
    features_nm=std_scaler.transform(features) 
    print("save scaler")
    f = open(pathsave + "train_merged_scaler.pickle", "wb")
    pickle.dump(std_scaler, f)
    f.close()

    label_onehot= keras.utils.to_categorical(label, num_classes=len(set(label)))
    classfier_dim=label_onehot.shape[1]
    classifier = Sequential()
    classifier.add(Dense(first_dim, activation='relu', input_dim=total_num_features))
    classifier.add(Dense(second_dim, activation='relu',))
    classifier.add(Dense(classfier_dim, activation='softmax')) 
    classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    classifier.fit(features_nm,label_onehot,epochs=epochs, batch_size=512,verbose=2)
    classifier.save(pathsave+ "keras.h5")   

#test model (keras) -----------------------------------------------------------------------------------------------------
def class_test(paht_testcsv,pathsave,features_test,label_test):
    #print("=================="+'start'+"==================")
    classifier = keras.models.load_model(pathsave+ "keras.h5")
    #t_vector = numpy.load("model/classifier_mfcc.npy")
    #test_data = numpy.load("model/" + 'pretrained_'+id + "_testdata.npy")
    f = open(pathsave+"train_merged_scaler.pickle", "rb")
    scaler = pickle.load(f)
    f.close()
    label_test_onehot= keras.utils.to_categorical(label_test, num_classes=len(set(label_test)))
    features_nm = scaler.transform(features_test)
    predict_output = classifier.predict(features_nm)
    loss_and_metrics = classifier.evaluate(features_nm, label_test_onehot, batch_size=128)
    print(loss_and_metrics)
    return(predict_output)

# train and test model (KNN) -----------------------------------------------------------------------------------------------------
def KNN_model(features,label,features_test,label_test):
    std_scaler=preprocessing.StandardScaler().fit(features)
    features_train=std_scaler.transform(features)   
    features_test_nm = std_scaler.transform(features_test)
    #label_onehot= keras.utils.to_categorical(label, num_classes=len(set(label)))
    KNN = KNeighborsClassifier(n_neighbors=len(set(label)))
    KNN.fit(features_train, label)
    predict_output = KNN.predict(features_test_nm)
    score= accuracy_score(predict_output, label_test)
    print(score)
    return(predict_output)
  


if __name__ == '__main__':   
    sample_rate = 8000
    nfft = 256
    frame_size = nfft
    frame_size_seconds = nfft/sample_rate
    frame_stride = int(frame_size/2)
    frame_stride_seconds = frame_size_seconds / 2
    silence_cutoff = 30
    num_cep = 20
    num_cep_useful = num_cep - 1
    total_num_features = num_cep_useful * 3       
    first_dim=18
    second_dim=6
    epochs=500

    pathwav ="data/" 
    outputpath ="csv/"  
    paht_test="test_data/"
    paht_testcsv="test_csv/"
    pathsave="model/"

    #1. run extract_mfcc() function
    #extract features for trainning data
    extract_mfcc(pathwav, outputpath)

    #extract features for test data
    extract_mfcc(paht_test, paht_testcsv)
    
    #2. run merged_train() function
    #train data: merge training data and create lable
    features,label = merged_train(outputpath,pathsave)
    #test data : merge test data and create lable (for )
    features_test,label_test = merged_train(paht_testcsv,pathsave)
    
 
    #3. run class_model() function 
    # train model by using Kears
    class_model(total_num_features,first_dim,second_dim,epochs,pathsave,features,label)
    
    #3. run class_test() function 
    # test data by using Kears
    test_output=class_test(paht_testcsv,pathsave,features_test,label_test)
    
    #4. run KNN_model() function
    # train and test KNN model by using sklearn
    test_output_knn=KNN_model(features,label,features_test,label_test)
    




