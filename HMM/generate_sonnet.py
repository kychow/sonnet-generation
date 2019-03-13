import os
import numpy as np
from Utility import Utility
from HMM import HiddenMarkovModel
from HMM import unsupervised_HMM
from HMM_helper import *
import numpy as np

rhyme_dict = Utility.create_rhyme_dict()
print(rhyme_dict)

# encoded_lines, words_dict = Utility.load_shakespeare_hidden()
#
# hmm = unsupervised_HMM(encoded_lines, 10, 3)
# for i in range(14):
#     print(sample_line(hmm, words_dict, n_syllables=10))
