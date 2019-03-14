import numpy as np
from keras.models import Sequential, load_model
from generate_lstm import indices_char_dict, indices_char_dict

model = create_model()
model = load_weights('lstm.h5')
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
