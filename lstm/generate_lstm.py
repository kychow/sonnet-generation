# implementation of a character-based LSTM to generate sonnets
import numpy as np
import random
import string
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Lambda
from keras.layers import LSTM
from keras.callbacks import LambdaCallback, EarlyStopping

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
        line = line.lstrip(' ').rstrip(' ')
        if line != '\n' and not line[0].isdigit():
            line.translate(str.maketrans('', '', string.punctuation))
            text += line.lower()

    # make char to index and index to char dictionary 
    characters = sorted(list(set(text)))
    char_indices_dict = dict((c, i) for i, c in enumerate(characters))
    indices_char_dict = dict((i, c) for i, c in enumerate(characters))
    #print(char_indices_dict)

    # makes every [step] char sequences of length seq_length and their outputs
    sequences = []
    next_chars = [] # next char that seq in sequences generates
    #print(repr(text[len(text) - 200:]))
    for i in range(0, len(text) - seq_length, step):
        #print(i, seq, text[i : i + seq_length])
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

def make_model(temperature=1.0):
    x, y, sequences, indices_char_dict, char_indices_dict, text = preprocess()
    model = Sequential()
    model.add(LSTM(200))
    # add temperature (controls variance)
    model.add(Lambda(lambda x: x / temperature))
    model.add(Dense(len(indices_char_dict), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto')
    model.fit(x, y, epochs=100, verbose=1, callbacks=[earlyStopping])
    model.save('lstm.h5')

def generate_sonnet():
    x, y, sequences, indices_char_dict, char_indices_dict, text = preprocess()

    model = load_model('lstm.h5')
    sonnet = []
    f = open('output.txt', 'a')

    seq = "shall i compare thee to a summer's day?\n"
    for _ in range(14):
        line = ""
        for i in range(40):
            x = np.zeros((1, len(seq), len(indices_char_dict)))
            for t, index in enumerate(seq):
                x[0, t, char_indices_dict[index]] = 1.

            prediction = model.predict(x, verbose=0)[0]
            index = np.argmax(prediction)
            char = indices_char_dict[index]
            line += char
            seq = seq[1:] + char

        sonnet.append(line)

    for line in sonnet:
        print(line)
        f.write(line)

make_model(0.25)
generate_sonnet()