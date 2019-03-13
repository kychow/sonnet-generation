# implementation of a character-based LSTM to generate sonnets
import numpy as np
import random
import string
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

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

x, y, sequences, indices_char_dict, char_indices_dict, text = preprocess()

model = Sequential()
#model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
# model.add(Dropout(0.5))
# standard fully-connected output layer w/ softmax linearity 
model.add(Dense(len(indices_char_dict), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop')
              #metrics=['accurjacy'])
model.fit(x, y, batch_size=32, epochs=10, verbose=1)
#score = model.evaluate(x_test, y_test, batch_size=16)

# make sure enough epochs that loss converges 
# draw softmax samples from trained model 
# mess around w/ temperature paremter (controls variance)

sonnet = []
for _ in range(1):

    # TODO define seed better 
    seed_index = random.randint(0, len(sequences) - 1)
    seq = text[seed_index:seed_index + len(sequences[0])] 
    print("Seed: " + seq)

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
        print(seq)
    
    sonnet.append(seq)

print(sonnet)