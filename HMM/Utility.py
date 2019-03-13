########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################
import string
import copy
import re

class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_lines():
        filename = "../data/shakespeare.txt"
        file = open(filename, "r")
        lines = []
        # translator = str.maketrans('', '', string.punctuation)
        for line in file:
            line = line.strip()
            if line != '' and not line[0].isdigit():
                words = line.split()

                # remove punctuation and capitalization
                for i in range(len(words)):
                    if words[i][-1] in [",", ".", "!", "?"]:
                        words[i] = words[i][:-1]
                    if words[i][-1] == "'" and words[i] not in ["th'", "t'"]:
                        words[i] = re.sub(r"[^\w\s-]" , '', words[i]).lower()
                    else:
                        words[i] = re.sub(r"[^\w'\s-]" , '', words[i]).lower()
                lines.append(words)
        return lines

    @staticmethod
    def load_shakespeare_hidden():
        lines = Utility.get_lines()
        words_dict = {}
        encoded_lines = copy.deepcopy(lines)
        count = 0
        for i in range(len(encoded_lines)):
            for j in range(len(encoded_lines[i])):
                word = lines[i][j]
                if word not in words_dict:
                    words_dict[word] = count
                    encoded_lines[i][j] = count
                    count += 1
                else:
                    encoded_lines[i][j] = words_dict[word]

        return encoded_lines, words_dict

    @staticmethod
    def create_rhyme_dict():
        rhyme_dict = {}
        lines = Utility.get_lines()
        # last_words = [line[-1] for line in lines]

        sonnet_length = 14
        # num_sonnets = int((len(lines)) / sonnet_length)
        num_sonnets = 154
        length_compensator = 0
        # for each sonnet
        s_start_i = 0
        s_end_i = sonnet_length
        for s in range(num_sonnets-1):
            print("sonnet {}".format(s+1))

            sonnet = lines[s_start_i:s_end_i]

            last_words = [line[-1] for line in sonnet]
            print(last_words)
            if s == 98:
                a_rhyme_pair = [last_words[0], last_words[2]]
                b_rhyme_pair = [last_words[1], last_words[3]]

                c_rhyme_pair = [last_words[5], last_words[7]]
                d_rhyme_pair = [last_words[6], last_words[8]]

                e_rhyme_pair = [last_words[9], last_words[11]]
                f_rhyme_pair = [last_words[10], last_words[12]]

                g_rhyme_pair = [last_words[13], last_words[14]]

                rhyme_pairs = [a_rhyme_pair, b_rhyme_pair, c_rhyme_pair, d_rhyme_pair,
                                e_rhyme_pair, f_rhyme_pair, g_rhyme_pair]

            elif s == 125:
                a_rhyme_pair = [last_words[0], last_words[1]]
                b_rhyme_pair = [last_words[2], last_words[3]]

                c_rhyme_pair = [last_words[4], last_words[5]]
                d_rhyme_pair = [last_words[6], last_words[7]]

                e_rhyme_pair = [last_words[8], last_words[9]]
                f_rhyme_pair = [last_words[10], last_words[11]]

                rhyme_pairs = [a_rhyme_pair, b_rhyme_pair, c_rhyme_pair, d_rhyme_pair,
                                e_rhyme_pair, f_rhyme_pair]

            else:
                print(last_words)
                a_rhyme_pair = [last_words[0], last_words[2]]
                b_rhyme_pair = [last_words[1], last_words[3]]

                c_rhyme_pair = [last_words[4], last_words[6]]
                d_rhyme_pair = [last_words[5], last_words[7]]

                e_rhyme_pair = [last_words[8], last_words[10]]
                f_rhyme_pair = [last_words[9], last_words[11]]

                g_rhyme_pair = [last_words[12], last_words[13]]

                rhyme_pairs = [a_rhyme_pair, b_rhyme_pair, c_rhyme_pair, d_rhyme_pair,
                                e_rhyme_pair, f_rhyme_pair, g_rhyme_pair]

            for rhyme_pair in rhyme_pairs:
                print(rhyme_pair)
                word_1 = rhyme_pair[0]
                word_2 = rhyme_pair[1]
                if word_1 not in rhyme_dict:
                    rhyme_dict[word_1] = []
                if word_2 not in rhyme_dict:
                    rhyme_dict[word_2] = []

                if word_2 not in rhyme_dict[word_1]:
                    rhyme_dict[word_1].append(word_2)
                if word_1 not in rhyme_dict[word_2]:
                    rhyme_dict[word_2].append(word_1)

            # 99th sonnet has 15 lines with rhyme pattern:
            # ababa cdcd efef gg
            # s_start_i = s * (sonnet_length) + length_compensator
            if s == 97:
                print("hi")
                s_start_i += sonnet_length
                sonnet_length = 15
                s_end_i += sonnet_length
            # 126th sonnet has 12 lines with rhyme pattern:
            # aabb ccdd eeff
            elif s == 124:
                s_start_i += sonnet_length
                sonnet_length = 12
                s_end_i += sonnet_length

            else:
                s_start_i += sonnet_length
                sonnet_length = 14
                s_end_i += sonnet_length

        print(rhyme_dict)
        return rhyme_dict
