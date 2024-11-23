'''
Created on 2024-10-14

@author: loson

Function:
1) load dataset
2) augmentation: add noise
3) extract features (MFCC, MEL Spectrogram, Zero Crossing Rate, Root Mean Square (RMS) Value, 
                     Chroma (chroma_stft), Tonnetz)
4) split dataset into train dataset and test dataset
'''

import os
import numpy as np
import librosa
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ravdess_dataset="E:\My Workspace\courses and assignments\COMP-5112-Research Methodology\project\dataset\Ravdess\\"
cafe_dataset="E:\My Workspace\courses and assignments\COMP-5112-Research Methodology\project\dataset\CaFE\\"

def load_ravdess(ravdess_path):
    ravdess_file_list = os.listdir(ravdess_path)
    emo_list=[]
    path_list=[]
    
    for tmp_pa in ravdess_file_list:
        member = os.listdir(ravdess_path + tmp_pa)
        
        for audio_f in member:
            f_name = audio_f.split('.')[0]
            eles = f_name.split('-')
            
            emo_list.append(int(eles[2]))
            path_list.append(ravdess_path + tmp_pa + '/' + audio_f)
            
    emo_df = pd.DataFrame(emo_list, columns=['Emotions'])
    pa_df = pd.DataFrame(path_list, columns=['Path'])
    data_loc = pd.concat([emo_df, pa_df], axis=1)
    
    # data_path.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 
    #                              6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    data_loc.to_csv("data_loc.csv",index=False)
    return data_loc

def add_noise(input_data):
    noise_crt = 0.030*np.random.uniform()*np.amax(input_data)
    output_data = input_data + noise_crt*np.random.normal(size=input_data.shape[0])
    return output_data

def extract_features(input_data, sample_rt):
    output_data = np.array([])

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=input_data, sr=sample_rt).T, axis=0)
    output_data = np.hstack((output_data, mfcc))
    # MelSpectogram
    melsp = np.mean(librosa.feature.melspectrogram(y=input_data, sr=sample_rt).T, axis=0)
    output_data = np.hstack((output_data, melsp))
    # # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=input_data).T, axis=0)
    output_data=np.hstack((output_data, zcr))
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=input_data).T, axis=0)
    output_data = np.hstack((output_data, rms))
    # Chroma_stft
    stft = np.abs(librosa.stft(input_data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rt).T, axis=0)
    output_data = np.hstack((output_data, chroma_stft))
    #Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(input_data), sr=sample_rt).T,axis=0)
    output_data = np.hstack((output_data, tonnetz))
    
    return output_data

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    noise_data=add_noise(data)
    res = extract_features(noise_data, sample_rate)
    result = np.array(res)
    
    return result

def split_data(test_size=0.2):
    
    ravdess_path=ravdess_dataset
    data_path=load_ravdess(ravdess_path)
    
    X, Y = [], []
    for path, emotion in zip(data_path.Path, data_path.Emotions):
        feature = get_features(path)
        X.append(feature)
        Y.append(emotion)
    
    x_train, x_test, y_train, y_test = train_test_split(np.array(X), Y, test_size=test_size, random_state=7)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
        
    return x_train, x_test, y_train, y_test


        

