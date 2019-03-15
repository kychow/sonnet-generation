# implementation of a character-based LSTM to generate sonnets
import numpy as np
import random
import string
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Lambda
from keras.layers import LSTM
from keras import callbacks 

def preprocess(filename="../data/shakespeare.txt", seq_length=40, step=5):
    '''
    returns semi-redundant sequences their outputs 

    seq_length: number of characters in each sequence
    step: gets every [step] sequence  
    '''

    # puts all data into text string  
    file = open(filename, "r")
    text = ""
    for line in file:
        line = line.strip()
        if line != '' and not line[0].isdigit():
            line.translate(str.maketrans('', '', string.punctuation))
            text += line

    # make char to index and index to char dictionary 
    characters = sorted(list(set(text)))
    char_indices_dict = dict((c, i) for i, c in enumerate(characters))
    indices_char_dict = dict((i, c) for i, c in enumerate(characters))

    # makes every [step] char sequences of length seq_length and their outputs
    sequences = []
    next_chars = [] # next char that seq in sequences generates
    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i : i + seq_length])
        next_chars.append(text[i + seq_length])

    # put sequences and outputs into np array
    x = np.zeros((len(sequences), seq_length, len(characters)))
    y = np.zeros((len(sequences), len(characters)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, char_indices_dict[char]] = 1
        y[i, char_indices_dict[next_chars[i]]] = 1

    return x, y, sequences, indices_char_dict, char_indices_dict, text

def make_model():
    x, y, sequences, indices_char_dict, char_indices_dict, text = preprocess()
    model = Sequential()
    model.add(LSTM(200))
    # add temperature (controls variance)
    # model.add(Lambda(lambda x: x / 1.5))
    model.add(Dense(len(indices_char_dict), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    earlyStopping = [callbacks.EarlyStopping(monitor='loss', verbose=0, mode='auto')]
    model.fit(x, y, epochs=5, verbose=1, callbacks=earlyStopping)
    model.save('lstm.h5')
    return indices_char_dict, char_indices_dict

def generate_sonnet():
    x, y, sequences, indices_char_dict, char_indices_dict, text = preprocess()
    model = load_model('lstm.h5')
    sonnet = []
    for _ in range(14):
        seq = "shall i compare thee to a summer's day? "
        line = ""
        for i in range(40):
            x = np.zeros((1, len(seq), len(indices_char_dict)))
            for t, index in enumerate(seq):
                x[0, t, char_indices_dict[index]] = 1

            prediction = model.predict(x, verbose=0)[0]
            index = np.argmax(prediction)
            char = indices_char_dict[index]
            line += char
            seq = seq[1:] + char

        sonnet.append(seq)

    for line in sonnet:
        print(line)
    
generate_sonnet()