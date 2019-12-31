def print_inputs():
	print("convert: X_train")
	print("sigmoid_CNN: flatten, x, neurons")
	print("softmax_CNN: flatten, x, neurons, final")
	print("embedding_CNN: vocab_size,max_length,x,neurons")
	print("vector: X_train, X_test")

def print_outputs():
	print("All CNN return model")
	print("Vector: X_train_pad, X_test_pad, vocab_size")
	
def print_descriptions():
	print("convert: Take text normalized")
	print("sigmoid_CNN: CNN with Sigmoid @end")
	print("softmax_CNN: CNN with Softmax @end")
	print("embedding_CNN: CNN for vectors")
	print("vector: Vectorize input")
	
def imports():
	print("import numpy as np")
	print("import pandas as pd")
	print("import tensorflow as tf")
	print("from tensorflow import keras")
	print("from keras.layers import Flatten, Dense, Dropout, Embedding")
	print("from keras.models import Sequential")

def convert(X_train):
	import re
	from bs4 import BeautifulSoup
	for i in range(len(X_train)):
		X_train[i] = BeautifulSoup(X_train[i], features="lxml")
		X_train[i] = re.sub("[^a-zA-z]", " ", X_train[i].get_text())
		X_train[i] = X_train[i].lower()

def sigmoid_CNN(flatten, x, neurons):
	model = Sequential()
	if(flatten == True):
		model.add(Flatten())
	for i in range(x+1):
		model.add(Dense(neurons[i], activation="relu"))
	model.add(Dense(1, activation="sigmoid"))
	return model

def softmax_CNN(flatten, x, neurons, final):
	model = Sequential()
	if(flatten == True):
		model.add(Flatten())
	for i in range(x+1):
		model.add(Dense(neurons[i], activation="relu"))
	model.add(Dense(final, activation="softmax"))
	return model

def embedding_CNN(vocab_size,max_length,x,neurons):
	model = Sequential()
	model.add(Embedding(vocab_size, input_length=max_length))
	for i in range(x+1):
		model.add(Dense(neurons[i], activation="relu"))
	model.add(Dense(1, activation="sigmoid"))
	return model

def vector(X_train, X_test):
	from numpy import array
	from keras.preprocessing.text import one_hot
	from keras.preprocessing.sequence import pad_sequences
	
	vocab_size = 500

	X_train = [one_hot(d, vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ') for d in X_train]
	X_test = [one_hot(d, vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ') for d in X_test]

	max_length = 25
	X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
	X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
