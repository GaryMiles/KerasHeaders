# KerasHeaders
from google.colab import drive

drive.mount('/content/drive')

!cp -r /content/drive/My\ Drive/Kaggle/Disaster\ NLP\ Model/Kaggle /content/Kaggle

!apt-get install -qq git

!git clone https://github.com/GaryMiles/KerasHeaders.git && mv KerasHeaders/headers.py /content/

-----------------------------------------------------


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
