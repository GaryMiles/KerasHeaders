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
		X_train[i] = BeautifulSoup(X_train[i])
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
	from tensorflow.python.keras.preprocessing.text import Tokenizer
	from tensorflow.python.keras.preprocessing.sequence import pad_sequences

	tokenizer_obj = Tokenizer()
	total = str(X_train + X_test)
	tokenizer_obj.fit_on_texts(str(total))

	max_length = max([len(s.split()) for s in total])
	vocab_size = len(tokenizer_obj.word_index) + 1

	X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
	X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)	

	X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
	X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

	return X_train_pad, X_test_pad, vocab_size
