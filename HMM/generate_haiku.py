import os
import numpy as np
from Utility import Utility
from HMM import HiddenMarkovModel
from HMM import unsupervised_HMM
from HMM_helper import *
import numpy as np
import random

syllable_map = Utility.get_words_to_syllables_dict()
rhyme_dict = Utility.create_rhyme_dict()
encoded_lines, words_dict = Utility.load_shakespeare_hidden()

hmm = unsupervised_HMM(encoded_lines, 6, 10)
haiku = []

haiku.append(sample_line(hmm, words_dict, syllable_map, n_syllables=5))
haiku.append(sample_line(hmm, words_dict, syllable_map, n_syllables=7))
haiku.append(sample_line(hmm, words_dict, syllable_map, n_syllables=5))

for line in haiku:
    print(line)
