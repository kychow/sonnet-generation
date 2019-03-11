########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import operator
import numpy as np
from numpy.random import choice
import math

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.

        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # initialize the second row of probs and seqs
        # trivial since the length-1 sequence of hidden states ending with self.O[i][x[0]]
        # that is most likely to have produced x1 = x[0] is just self.O[i][x[0]]
        for j in range(0, self.L):
            # x[0] is obervation at 1-1 = 0
            probs[1][j] = self.A_start[j] * self.O[j][x[0]]

        # for each observation
        for i in range(2, M+1):
            # for each state
            for j in range(0, self.L):
                state_probs = []
                for k in range(0, self.L):
                    # # observation at i-1
                    # x_i = x[i-1]
                    state_prob = probs[i-1][k] * self.O[j][x[i-1]] * self.A[k][j]
                    state_probs.append(state_prob)

                # Append state j's max prob and corresponding sequence
                # max_prob_i is most probable state
                max_prob_i, max_prob = max(enumerate(state_probs), key=operator.itemgetter(1))
                probs[i][j] = max_prob
                seqs[i][j] = seqs[i-1][max_prob_i] + str(max_prob_i)

        # get max prob of last obersvation and corresponding sequence
        max_prob_i, max_prob = max(enumerate(probs[M]), key=operator.itemgetter(1))
        max_seq = seqs[len(probs)-1][max_prob_i] + str(max_prob_i)
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        # Initialize alphas as zeros
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # initialize the second row of alphas
        for j in range(0, self.L):
            alphas[1][j] = self.A_start[j] * self.O[j][x[0]]
        if normalize:
            norm_const = sum(alphas[1])
            for j in range(0, self.L):
                alphas[1][j] /= norm_const

        # for each observation
        for i in range(2, M+1):
            # for each state
            for j in range(0, self.L):
                # for each state
                for k in range(0, self.L):
                    alphas[i][j] += (alphas[i-1][k] * self.A[k][j])
                alphas[i][j] *= self.O[j][x[i-1]]

            if normalize:
                norm_const = sum(alphas[i])
                for j in range(0, self.L):
                    alphas[i][j] /= norm_const
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # initialize the second row of betas
        for j in range(0, self.L):
            betas[M][j] = 1

        # for each observation
        for i in range(M-1, 0, -1):
            # for each state
            for j in range(0, self.L):
                # for each state
                for k in range(0, self.L):
                    betas[i][j] += (betas[i+1][k] * self.A[j][k] * self.O[k][x[i]])

            if normalize:
                norm_const = sum(betas[i])
                for j in range(0, self.L):
                    betas[i][j] /= norm_const
        return betas

    def M_step_transitions(self, a, b, X, Y):
        '''
        Counts the number of transitions where Y[i][j-1] = a
        and the number of transitions where  Y[i][j-1] = a and Y[i][j] = b
        and returns the ratio
        '''
        num_a = 0
        num_trans = 0
        for i in range(0, len(X)):
            for j in range(1, len(X[i])):
                if Y[i][j-1] == a:
                    num_a += 1
                    if Y[i][j] == b:
                        num_trans += 1
        result = num_trans / num_a
        return result

    def M_step_observations(self, w, a, X, Y):
        '''
        Counts the number of transitions where Y[i][j] = a
        and the number of transitions where  X[i][j] = w
        and returns the ratio
        '''
        num_a = 0
        num_obs = 0
        for i in range(0, len(X)):
            for j in range(0, len(X[i])):
                if Y[i][j] == a:
                    num_a += 1
                    if X[i][j] == w:
                        num_obs += 1
        result = num_obs / num_a
        return result


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                self.A[a][b] = self.M_step_transitions(a, b, X, Y)

        # Calculate each element of O using the M-step formulas.
        for a in range(len(self.O)):
            for w in range(len(self.O)):
                self.O[a][w] = self.M_step_observations(a, w, X, Y)


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for iter in range(0, N_iters):
            A_nums = [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_nums = [[0. for _ in range(self.D)] for _ in range(self.L)]

            A_denoms = [0. for _ in range(self.L)]
            O_denoms = [0. for _ in range(self.L)]

            print(iter)
            # print(self.O)
            # print(self.A)

            for seq in X:
                # INIT
                M = len(seq)
                # numerators
                marg_prob_a = [[0. for _ in range(self.L)] for _ in range(M + 1)]
                marg_prob_ab = [[[0. for _ in range(self.L)] for j in range(self.L)] for _ in range(M + 1)]

                # denomenators
                marg_prob_a_denoms = [0. for _ in range(M + 1)]
                marg_prob_ab_denoms = [0. for _ in range(M + 1)]

                # E step
                alphas = self.forward(seq, normalize=True)
                betas = self.backward(seq, normalize=True)

                # print(np.array(alphas))
                # print(np.array(betas))
                # raise Exception

                for i in range(1, M+1):
                    norm_const = np.sum(alphas[i][k] * betas[i][k] for k in range(self.L))
                    marg_prob_a_denoms[i] += norm_const
                    for j in range(self.L):
                        marg_prob_a[i][j] += (alphas[i][j] * betas[i][j])

                for i in range(1, M):
                    denom = 0
                    for a in range(0, self.L):
                        for b in range(0, self.L):
                            denom += (alphas[i][a] * self.O[b][seq[i+1-1]] * self.A[a][b] * betas[i+1][b])
                    marg_prob_ab_denoms[i] += denom
                    for j in range(0, self.L):
                        for k in range(0, self.L):
                            num = alphas[i][j] * self.O[k][seq[i+1-1]] * self.A[j][k] * betas[i+1][k]
                            marg_prob_ab[i][j][k] += num

                for i in range(1, M+1):
                    for j in range(0, self.L):
                        marg_prob_a[i][j] /= marg_prob_a_denoms[i]

                for i in range(1, M):
                    for j in range(0, self.L):
                        for k in range(0, self.L):
                            marg_prob_ab[i][j][k] /= marg_prob_ab_denoms[i]

                # M step
                for j in range(0, self.L):
                    for k in range(0, self.L):
                        num = sum(marg_prob_ab[i][j][k] for i in range(1, M))
                        A_nums[j][k] += num
                    denom = sum(marg_prob_a[i][j] for i in range(1, M))
                    A_denoms[j] += denom

                for j in range(0, self.L):
                    for k in range(0, self.D):
                        num = sum(marg_prob_a[i][j] for i in range(1, M+1) if seq[i-1] == k)
                        O_nums[j][k] += num
                    denom = sum(marg_prob_a[i][j] for i in range(1, M+1))
                    O_denoms[j] += denom

            for j in range(0, self.L):
                for k in range(0, self.L):
                    self.A[j][k] = A_nums[j][k] / A_denoms[j]

            for j in range(0, self.L):
                for k in range(0, self.D):
                    self.O[j][k] = O_nums[j][k] / O_denoms[j]

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        start_states = [i for i in range(self.L)]
        state1 = choice(start_states, p=self.A_start)
        states.append(state1)
        p_val_emissions = self.O[states[-1]]
        emission.append(choice([i for i in range(self.D)], p=p_val_emissions))
        for i in range(1, M):
            p_val_states = self.A[states[-1]]
            states.append(choice([i for i in range(self.L)], p=p_val_states))
            p_val_emissions = self.O[states[-1]]
            emission.append(choice([i for i in range(self.D)], p=p_val_emissions))

        return emission, states

    
    def generate_sonnet_line(words_syllables_dict, obs_map_r, n_syllables):
        '''
        Generates an emission with M syllables, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            n_syllables:          number of syllables of the emission to generate.

        Returns:
            words:   The randomly generated emission as a list.
        '''

        states = []
        words = []
        syllables = 0

        start_states = [i for i in range(self.L)]
        state1 = choice(start_states, p=self.A_start)
        states.append(state1)
        p_val_emissions = self.O[states[-1]]
        emis = choice([i for i in range(self.D)], p=p_val_emissions)
        word = obs_map_r[emis]
        words.append(word)

        while syllables < n_syllables: 
            p_val_states = self.A[states[-1]]
            states.append(choice([i for i in range(self.L)], p=p_val_states))
            p_val_emissions = self.O[states[-1]]
            emis = choice([i for i in range(self.D)], p=p_val_emissions)

            word = obs_map_r[emis]
            syllables += words_syllables_dict[word]
            words.append(word)

        return words 

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)
    random.seed(2019)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM