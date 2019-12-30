def imports():
  import numpy as np 
  import pandas as pd 
  import tensorflow as tf
  from tensorflow import keras
  from keras.layers import Flatten, Dense, Dropout, Embedding
  from keras.models import Sequential

def convert(X_train):
  import re
  from bs4 import BeautifulSoup
  for i in range(len(X_train)):
    X_train[i] = BeautifulSoup(X_train[i]) 
    X_train[i] = re.sub("[^a-zA-Z]”, " ", X_train[i].get_text()) 
    X_train[i] = X_train[i].lower()

def sigmoid_CNN(flatten, x, neurons):
  model = Sequential()
  if(flatten == True):
    model.add(Flatten())
  for i in range(x+1):
    model.add(Dense(neurons[i], activation="relu"))
  model.add(Dense(1, activation=“sigmoid"))
  return model

def softmax_CNN(flatten, x, neurons, final):
  model = Sequential()
  if(flatten == True):
    model.add(Flatten())
  for i in range(x+1):
    model.add(Dense(neurons[i], activation="relu"))
  model.add(Dense(final, activation="softmax"))
  return model

def embedding_CNN(vocab_size, max_length, x, neurons):
  model = Sequential()
  model.add(Embedding(vocab_size, input_length=max_length))
  for i in range(x+1):
    model.add(Dense(neurons[i], activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  return model
