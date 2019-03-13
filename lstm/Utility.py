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
    def load_shakespeare_hidden():
        filename = "../data/shakespeare.txt"
        file = open(filename, "r")
        lines = []
        words_dict = {}
        # lines_numbered = []
        translator = str.maketrans('', '', string.punctuation)
        for line in file:
            line = line.strip()
            if line != '' and not line[0].isdigit():
                words = line.split()

                # remove punctuation and capitalization
                for i in range(len(words)):
                    if words[i][-1] in [",", ".", "!", "?"]:
                        words[i] = words[i][:-1]
                    if words[i][-1] == "'"  and words[i] not in ["th'", "t'"]:
                        words[i] = re.sub(r"[^\w\s-]" , '', words[i]).lower()
                    else:
                        words[i] = re.sub(r"[^\w'\s-]" , '', words[i]).lower()
                lines.append(words)

        lines_numbered = copy.deepcopy(lines)
        count = 0
        for i in range(len(lines_numbered)):
            for j in range(len(lines_numbered[i])):
                word = lines[i][j]
                if word not in words_dict:
                    words_dict[word] = count
                    lines_numbered[i][j] = count
                    count += 1
                else:
                    lines_numbered[i][j] = words_dict[word]

        return lines_numbered, words_dict
