import labeler.models.tag_manager as tm
import labeler.models.word_bag as wb
import numpy as np


def classify_file(label_dict):
    # Get classification starting percentages from word bag
    classify_dict = {}
    bag = wb.WordBag()
    for row, column_dict in label_dict[tm.CONTENT].items():
        classify_dict[row] = {}
        for column, word_list in column_dict.items():
            classify_dict[row][column] = []
            for word in word_list:
                try:
                    clean_word = bag.clean_digits(word['word'])
                    word['probabilities'] = bag.label_probabilities(clean_word)
                except (TypeError, IndexError):
                    continue
                classify_dict.append(word)

        # For each row, apply the bigram classification algorithm


# def test(word1, word2):
#     bag = wb.WordBag()

#     word1_prob_iter = map(lambda x: x[1], bag.raw_label_probabilities(word1))
#     word1_array = np.fromiter(word1_prob_iter, np.float)
#     label_count = word1_array.shape[0]
#     word2_prob_iter = map(lambda x: x[1], bag.raw_label_probabilities(word2))
#     word2_array = np.fromiter(word2_prob_iter, np.float)

#     bigram_prob_iter1 = map(lambda x: x[1], bag.raw_bigram_probabilities_1_given_2())
#     bigram_prob_iter2 = map(lambda x: x[1], bag.raw_bigram_probabilities_2_given_1())
#     bigram_array1 = np.fromiter(bigram_prob_iter1, np.float).reshape(-1, label_count)
#     bigram_array2 = np.fromiter(bigram_prob_iter2, np.float).reshape(-1, label_count)

#     prod_array1 = (bigram_array2 * word1_array).T
#     prod_array2 = bigram_array1 * word2_array

#     fullprint(word1_array, 'word1_array')
#     fullprint(bigram_array2, 'bigram_array2')
#     fullprint(prod_array1, 'prod_array1')
#     fullprint(prod_array2, 'prod_array2')

#     sum_array = prod_array1 + prod_array2
#     # print(sum_array)
#     index = np.unravel_index(sum_array.argmax(), sum_array.shape)
#     print((index[0] + 1, index[1] + 1))
#     print(sum_array[index[0], index[1]])
#     print(sum_array[2, 4])
#     print(sum_array[4, 2])


# def fullprint(array, filename='dump.txt'):
#     opt = np.get_printoptions()
#     np.set_printoptions(threshold=np.nan)
#     with open(filename, 'w') as dumpfile:
#         dumpfile.write(np.array_str(array))
#     np.set_printoptions(**opt)
#     return


# --------------------------------------------------------------------------


def word_probabilities(bag, *words):
    """Generates probabilities for each word

    For each word returns probability of being in each class based on the
    word bag and relative positions (bigrams) of the words.
    """
    # First create a list of all the probability arrays for each word
    word_arrays = []
    for word in words:
        word_arrays.append(monogram_probabilites(bag, word))

    # Now we apply the bigram probability algorithm
    bigram_array = []
    for first, second in zip(word_arrays, word_arrays[1:]):
        # First and second are the np.arrays with classification probabilities
        bigram_array.append(bigram_probabilities(bag, first, second))
    bigram_sequence(bigram_array)
    return bigram_array


def monogram_probabilites(bag, word):
    """Get an array of smoothed probabilities for an individual word

    These values at this time are only based upon the probabilies as found
    in the word bag. The probabilities are smoothed so each class is always
    possible. Uses Laplace smoothing with a pseudocount value of (1/C) where
    C is the number of possible different classes.

    Returns:
      np.array: 1D array of length C where each element is the probability
        of classifying the word as the class whose id is the corresponding
        index of the probability in the array
    """
    word_prob_iter = map(lambda x: x[1], bag.raw_label_probabilities(word))
    return np.fromiter(word_prob_iter, np.float)


def bigram_probabilities(bag, first, second):
    """Get an array of smoothed probabilities for bigrams

    Uses the monogram probabilities to modify the bigram probabilities in
    in the following way:
      - Get the smoothed probabilities of all possible bigrams (with the
          same Laplace smoothing as described in the monogram_probabilities
          docsting)
      - Calculate the estimated probabilities of the first word
          classifications using the bigram probability multiplied by the
          actual observed probability of the following word
      - Transpose the bigram probabilities so it now gives the probabilities
          of the second word classification given the first word
      - Calculate the estimated probabilities of the second word
          classifications using the bigram probability multiplied by the
          actual observed probability of the preceding word
      - Sum the estimated classification probabilities for the first and
          second word to get the bigram classification probabilities


    Args:
      first (np.array): 1D array of classification probabilities
      second (np.array): 1D array of classification probabilities

    Returns:
      np.array: 2D array of bigram probabilities where rows represent the
        classification of the first word and columns the second
    """
    # Validate the shapes of the first and second arrays
    if not all_equal(first.shape, second.shape):
        raise ValueError('Argument arrays not all the same size')

    # Gets the number of possible labels from the first array shape
    label_count = first.shape[0]

    # Get the bigram array from the word bag
    bigram_prog_iter = map(lambda x: x[1], bag.raw_bigram_probabilities())
    bigram_array = np.fromiter(bigram_prog_iter, np.float)
    bigram_array = bigram_array.reshape(-1, label_count)

    # Calculate the estimated first word probabilities based on the second
    est_first_prob = bigram_array * second
    est_second_prob = bigram_array.T * first

    # Return the sum of the estimated probability arrays
    return est_first_prob + est_second_prob


def bigram_sequence(bigrams):
    """Get the most probable sequence of bigrams

    Args:
      bigrams (List[np.array]): list of 2D arrays, where each array has
        probabilities of each sequential bigram classifications
    """
    # Validate all bigram arrays are the same shape
    if not all_equal([array.shape for array in bigrams]):
        raise ValueError('All bigram arrays must be the same shape')

    last = np.zeros(bigrams[0].shape)
    for array in bigrams[::-1]:
        np.log(array) + last


def all_equal(*items):
    return all(items[0] == item for item in items)


if __name__ == '__main__':
    TEST_STRING = '302 N'
    bag = wb.WordBag()
    bigrams = word_probabilities(bag, *TEST_STRING.split())
    print(len(bigrams))

    for i in range(len(bigrams)):
        index = np.unravel_index(bigrams[i].argmax(), bigrams[i].shape)
        print((index[0] + 1, index[1] + 1))
