import os
import numpy as np
import Utility

encoded_lines, words_dict = Utility.load_shakespeare_hidden()

hmm = unsupervised_HMM(encoded_lines, 10, 100)
for i in range(14):
    print(sample_line(hmm, words_dict, n_syllables=10))
