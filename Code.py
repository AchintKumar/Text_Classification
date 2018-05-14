#Libraries
from __future__ import print_function
import tensorflow as tf
import os
#import sys
import numpy as np
import pandas as pd 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


#Setting path
print(os.getcwd())
os.chdir('/media/achint/INSOFE/Cute5')

#Reading test data
test_raw=pd.read_csv('test.csv')
print(test_raw.dtypes)
#Changing to string
test_raw=test_raw.astype('str')
print(test_raw.isnull().any())

#Reading train data
train_raw=pd.read_csv('train.csv')
train_raw=train_raw.astype('str')
#Taking number of words for tokenizing
token=Tokenizer(num_words=5000)
token.fit_on_texts(train_raw.converse)

#Taking top 50 length of words
train_x=token.texts_to_sequences(train_raw.converse)
train_x=sequence.pad_sequences(train_x, maxlen=50)
test_x=token.texts_to_sequences(test_raw.converse)
test_x=sequence.pad_sequences(test_x, maxlen=50)

#Converting labels to categorical integers
labels=list(train_raw.categories.unique())
print(labels)
train_y = np.array([labels.index(i) for i in train_raw.categories])
train_y = to_categorical(train_y)



#LSTM model
model=Sequential()
model.add(Embedding(5000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_x, train_y,batch_size=32,epochs=2)
#predicting model
pred_1=model.predict_classes(test_x)
print(pred_1)
print(labels)
#converting integers back into labels
results=[]
for i in pred_1:
	results.append(labels[i])
print(results)
#Writing files for submission
results=pd.DataFrame(results)
results.to_csv('prediction.csv')


#LSTM Model 2
model=Sequential()
model.add(Embedding(5000, 128))
model.add(LSTM(128, input_shape=(50,1),return_sequences=True))
model.add(LSTM(128, input_shape=(50,1),return_sequences=False))
model.add(Dense(50))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])
model.fit(train_x, train_y,batch_size=32,epochs=2)

pred_2=model.predict_classes(test_x)
print(pred_2)
print(labels)
results_2=[]
for i in pred_2:
	results_2.append(labels[i])
print(results_2)
results_2=pd.DataFrame(results_2)
results_2.to_csv('pred2.csv')
