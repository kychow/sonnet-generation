import os
import numpy as np
from Utility import Utility
from HMM import HiddenMarkovModel
from HMM import unsupervised_HMM
from HMM_helper import *
import numpy as np
import random

rhyme_dict = Utility.create_rhyme_dict()
# print(rhyme_dict)
encoded_lines, words_dict = Utility.load_combined_hidden()

hmm = unsupervised_HMM(encoded_lines, 6, 10)
sonnet = [''] * 14

rhyme_scheme = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 13]]
for pair in rhyme_scheme:
    rhyme_word_1, rhyme_word_2 = random.choice(list(rhyme_dict.items()))
    encoded_rhyme_word_1 = words_dict[rhyme_word_1]
    encoded_rhyme_word_2 = words_dict[rhyme_word_2[0]]
    sonnet[pair[0]] = sample_line_combined_rhyme(hmm, encoded_rhyme_word_1, words_dict, n_syllables=10)
    sonnet[pair[1]] = sample_line_combined_rhyme(hmm, encoded_rhyme_word_2, words_dict, n_syllables=10)

for line in sonnet:
    print(line)
