import librosa

signal,sr =librosa.load(pathwav+f, mono=True,sr=sample_rate)
signal_trimed, index = librosa.effects.trim(signal, top_db=silence_cutoff)

import os
import pandas
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank

librosa.feature.mfcc(y=signal_trimmed, sr=sr, n_mfcc=13)

#-----------MFCC-----------------
MFCC = mfcc(signal_trimed,  sample_rate, winlen=frame_size_seconds, winstep = frame_stride_seconds, winfunc=numpy.hamming, nfft=nfft, numcep = num_cep)
# do not use the first MFCC coef
MFCC = MFCC[:, 1:num_cep]
# make sure MFCC and pitches have the same number of frames 
# as they come from different library which does different things in the end of the singal
names = ['MFCC']*num_cep_useful
features= MFCC  