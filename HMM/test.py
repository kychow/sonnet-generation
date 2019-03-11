import os
import numpy as np
from Utility import Utility
from HMM import HiddenMarkovModel
from HMM import unsupervised_HMM
from HMM_helper import *
import numpy as np
# from HMM_helper import (
#     text_to_wordcloud,
#     states_to_wordclouds,
#     parse_observations,
#     sample_sentence,
#     sample_line,
#     visualize_sparsities,
#     animate_emission
# )

encoded_lines, words_dict = Utility.load_shakespeare_hidden()

hmm = unsupervised_HMM(encoded_lines, 10, 10)
for i in range(14):
    print(sample_line(hmm, words_dict, n_syllables=10))
