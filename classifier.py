import collections
import itertools
import numpy as np
import scipy.spatial as spspatial
import labeler.models.tag_manager as tm
import labeler.models.word_bag as wb
import labeler.models.config as cfg


cfg.Config.initialize_config()


class NetworkInput:

    """Object that will be passed to the NN for further classification"""
    # TODO:  Separate this into a separate modules?

    LENGTH_CLASSES = 10

    @classmethod
    def vector_length(cls):
        """Gets the length the output vector will be"""
        length = 0
        length += len(cfg.Config.labels) * 3  # Label probs for each word
        length += len(cfg.Config.headers)  # Encoded headers
        length += len(cfg.Config.punctuation) * 2  # Left and right punctuation
        length += cls.LENGTH_CLASSES  # Length classes vector
        return length

    @classmethod
    def inputs_from_tag_cell(cls, tag_cell, header):
        """Returns a list of NetworkInput instances for the tag_cell"""
        word_list = tag_cell.split
        # Add cell beginning and ending information
        word_list.insert(0, '')
        word_list.insert(0, np.zeros((len(cfg.Config.labels),)))
        word_list.append('')
        word_list.append(np.zeros((len(cfg.Config.labels),)))

        # Iterate the word list in groups of 5 (left, punc, word, punc, right)
        # TODO:  I'm tired... better way of doing this (I'm sure there is)
        network_inputs = []
        for l_word, l_punc, word, r_punc, r_word in [word_list[i:i + 5] for i in range(0, len(word_list) - 4, 2)]:
            try:
                l_probs = l_word.tag_probabilities
            except AttributeError:
                l_probs = l_word
            try:
                r_probs = r_word.tag_probabilities
            except AttributeError:
                r_probs = r_word

            net_input = cls(
                word, l_probs, r_probs, l_punc, r_punc, header
            )
            network_inputs.append(net_input.vectorize())
        return network_inputs

    def __init__(
            self, tag_word, left_word_probs, right_word_probs,
            left_punc, right_punc, header, label=None):
        """Represent the input to the neural network

        Args:
          header (int): Index of the desired header (zero-indexed)
        """
        # Use tag_probabilities on the tag word objects
        self.tag_word = tag_word
        self.left_word_probs = left_word_probs
        self.right_word_probs = right_word_probs
        self.left_punc = left_punc
        self.right_punc = right_punc
        self.header = header
        # TODO:  Just pull this from the tag_word instance?
        self.label = label  # If not None, we are training

        # print('\nInitializing NetworkInput:')
        # print('         tag_word : {}'.format(repr(self.tag_word)))
        # print('  left_word_probs : {}'.format(self.left_word_probs))
        # print(' right_word_probs : {}'.format(self.right_word_probs))
        # print('        left_punc : "{}"'.format(self.left_punc))
        # print('       right_punc : "{}"'.format(self.right_punc))
        # print('           header : {}'.format(self.header))

        self.left_punc_list = self.encode_punc(self.left_punc)
        self.right_punc_list = self.encode_punc(self.right_punc)
        self.header_list = self.encode_header(self.header)
        self.length_list = self.encode_length(str(self.tag_word))

        # Other inputs:
        #  Contains numerals?

    def vectorize(self):
        return np.fromiter(
            itertools.chain(
                self.tag_word.tag_probabilities,
                self.left_word_probs,
                self.right_word_probs,
                self.header_list,
                self.left_punc_list,
                self.right_punc_list,
                self.length_list
            ), np.float64
        )

    def encode_punc(self, string):
        """Get a list that represents which punctuation is in the string"""
        # TODO:  Check if this is faster with numpy that regular lists
        punc_list = [0] * len(cfg.Config.punctuation)
        for index, punc_str in enumerate(cfg.Config.punctuation.values()):
            if punc_str in string:
                punc_list[index] += 1  # TODO:  Check assign or add faster
        return punc_list

    def encode_header(self, header):
        """Get a list that represents the given header"""
        header_list = [0] * len(cfg.Config.headers)
        header_list[header] += 1
        return header_list

    def encode_length(self, string):
        """Return list indicating string length class

        Possible classes are each individual length 1-9 and 10+
        """
        length_list = [0] * self.LENGTH_CLASSES
        try:
            length_list[(len(string) - 1)] += 1
        except IndexError:
            length_list[(self.LENGTH_CLASSES - 1)] += 1
        return length_list

    def encode_label(self, label):
        """Get a list that represents the given label (for training)"""
        label_list = [0] * len(cfg.Config.labels)
        label_list[label] += 1
        return label_list


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


def classify_tag_manager(tag_manager):
    bag = wb.WordBag()

    label_count = collections.defaultdict(int)  # key = (column, label)
    for index in tag_manager.index_iterator():
        cell = tag_manager.get(*index)
        if cell is None:
            continue
        # Otherwise cell is a tm.TagCell instance
        classify_cell(cell, label_count, bag=bag)
    header_map = {res['id'] - 1: res['name'] for res in bag.get_headers()}
    header_arrays, header_columns = header_prob_array(label_count, bag=bag)
    best_header_indicies = best_header(header_arrays, bag=bag)
    final_headers = [header_map[head_idx] for head_idx in best_header_indicies]
    # with open('header_array.npy', 'wb') as array_dump:
    #     np.save(array_dump, header_arrays)
    # print(final_headers)

    if tag_manager.headers is None:
        tag_manager.headers = {}

    header_count = collections.defaultdict(int)
    for column, header in enumerate(final_headers):
        header_count[header] += 1
        if column in tag_manager.headers:
            tag_manager.headers[column]['tags'] = [header]
        else:
            tag_manager.headers[column] = {
                'tags': [header],
                'value': '{}{}'.format(header, header_count[header])
            }

    return best_header_indicies


def classify_cell(tag_cell, label_count, bag=None):
    """Takes word list a single cell in parsed dict in input words"""
    print('\nClassifying cell: ({cell.row}, {cell.column})'.format(cell=tag_cell))

    if bag is None:
        bag = wb.WordBag()
    labels = [x['name'] for x in bag.get_labels()]

    words = list(tag_cell.iter_word_strings())
    word_prob_list = word_probabilities_list([str(x) for x in words], bag=bag)
    for word, probs in zip(words, word_prob_list):
        word.tag_probabilities = probs

        best_label = probs.argmax()
        print('{: >15} : {}'.format(str(word), labels[best_label]))
        label_count[(word.column, best_label)] += 1


def header_prob_array(header_dict, bag=None):
    """Of format {(column, label): count}"""
    if bag is None:
        bag = wb.WordBag()

    label_count = len(bag.get_labels())
    column_dict = {}
    for (column, label), count in header_dict.items():
        if column not in column_dict:
            column_dict[column] = np.zeros((label_count,))
        column_dict[column][label] += count

    count_arrays = []
    column_indicies = []
    for column, count_array in sorted(column_dict.items()):
        count_arrays.append(count_array / count_array.sum())
        column_indicies.append(column)
    return (np.stack(count_arrays), column_indicies)


def word_probabilities_list(words, bag=None):
    if bag is None:
        bag = wb.WordBag()

    return word_probabilities(bag, *words)


def fullprint(array, filename='dump.txt'):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.nan)
    with open(filename, 'w') as dumpfile:
        dumpfile.write(np.array_str(array))
    np.set_printoptions(**opt)
    return


# --------------------------------------------------------------------------


def word_probabilities(bag, *words):
    """Generates probabilities for each word

    For each word returns probability of being in each class based on the
    word bag and relative positions (bigrams) of the words.
    """
    # First create a list of all the probability arrays for each word
    word_arrays = []
    for word in words:
        word_arrays.append(monogram_probabilites(bag, bag.clean_digits(word.upper())))

    if len(words) <= 1:
        # No possible bigrams
        return word_arrays

    # Now we apply the bigram probability algorithm
    bigram_array = []
    for first, second in zip(word_arrays, word_arrays[1:]):
        # First and second are the np.arrays with classification probabilities
        bigram_array.append(bigram_probabilities(bag, first, second))
    word_probs = bigram_sequence2(bigram_array)
    return word_probs


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
    return np.fromiter(word_prob_iter, np.float64)


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
    bigram_array_first_cond_second = np.fromiter(
        map(lambda x: x['probability'], bag.bigram_prob_first_cond_second()), np.float64
    ).reshape(-1, label_count)
    bigram_array_second_cond_first = np.fromiter(
        map(lambda x: x['probability'], bag.bigram_prob_second_cond_first()), np.float64
    ).reshape(-1, label_count)
    # bigram_prog_iter = map(lambda x: x[1], bag.raw_bigram_probabilities())
    # bigram_array = np.fromiter(bigram_prog_iter, np.float64)
    # bigram_array = bigram_array.reshape(-1, label_count)

    # fullprint(bigram_array_first_cond_second, 'bigram_array_first_cond_second.txt')
    # fullprint(bigram_array_second_cond_first, 'bigram_array_second_cond_first.txt')
    # fullprint(second, 'second.txt')

    # print(
    #     'bigram_array_first_cond_second: column sums = {}'.format(
    #         bigram_array_first_cond_second.sum(axis=0)
    #     )
    # )
    # print(
    #     'bigram_array_second_cond_first: column sums = {}'.format(
    #         bigram_array_second_cond_first.sum(axis=0)
    #     )
    # )

    # Calculate the estimated first word probabilities based on the second
    # fullprint(bigram_array, 'bigram_array.txt')
    # fullprint(first, 'first.txt')
    # fullprint(second, 'second.txt')
    est_first_prob = bigram_array_first_cond_second * second
    # fullprint(est_first_prob, 'est_first_prob.txt')
    # index = np.unravel_index(est_first_prob.argmax(), est_first_prob.shape)
    # print(index)
    # print(est_first_prob[index])
    est_second_prob = (bigram_array_second_cond_first * first).T
    # fullprint(est_second_prob, 'est_second_prob.txt')
    # index = np.unravel_index(est_second_prob.argmax(), est_second_prob.shape)
    # print(index)
    # print(est_second_prob[index])

    # print('est_first_prob: {}'.format(est_first_prob.sum()))
    # print('est_second_prob: {}'.format(est_second_prob.sum()))

    # Return the sum of the estimated probability arrays
    ret_array = (est_first_prob + est_second_prob) / 2

    # print('ret_array [0]: {}'.format(ret_array.sum(axis=0)))
    # print('ret_array [1]: {}'.format(ret_array.sum(axis=1)))
    # fullprint(ret_array, 'ret_array.txt')

    return ret_array


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
    path = []
    for array in bigrams[::-1]:
        last, index_array = sum_arrays(np.log(array), last)
        path.insert(0, index_array)

    index = np.unravel_index(last.argmax(), last.shape)
    print(index)
    # print(path)

    first = index[0]
    last = index[1]
    for i, test in enumerate(bigrams):
        print('[{}] Best bigram  : {}'.format(i + 1, (first, last)))
        next_last = path[i][first, last]
        first = last
        last = next_last
        index = np.unravel_index(test.argmax(), test.shape)
        print('    Original best:  {}'.format(index))


def bigram_sequence2(bigrams):
    # Validate all bigram arrays are the same shape
    if not bigrams:
        raise ValueError('Must have at least 1 bigram array')
    if not all_equal([array.shape for array in bigrams]):
        raise ValueError('All bigram arrays must be the same shape')

    word_probs = []
    last_array = None
    for word_array in bigrams:
        # Sum axis=1 for first probabilities, axis=0 for second
        first_prob = word_array.sum(axis=1)
        second_prob = word_array.sum(axis=0)
        if last_array is None:
            word_probs.append(first_prob)
            last_array = second_prob
        else:
            word_probs.append((last_array + first_prob) / 2.0)  # Faster than np.mean
            last_array = second_prob
    word_probs.append(last_array)
    return word_probs


def sum_arrays(array1, array2):
    """Sum two arrays: each column element summed with each row row element

    This method can be though of as placing the two 2D arrays on the faces of
    a cube and filling in the cube with the sum of the two elements that
    intersect at that point. We rotate the arrays so that each row of the
    first array has it's elements summed with each element of the column of the
    second array with the corresponding index.

    TODO:  Find a way to represent this with some ascii art for better
      visualization (I will forget)
    """
    # Check that the arrays are the proper size
    if not all_equal([array1.shape, array2.shape]):
        raise ValueError('All bigram arrays must be the same shape')
    if array1.ndim != 2 or array1.shape[0] != array1.shape[1]:
        raise ValueError('Arrays must be 2D squares')

    # Get the dimension of each array side
    dim = array1.shape[0]

    # First we broadcast the arrays to 3D arrays
    cube_array1 = np.broadcast_to(array1, (dim,) * 3)
    cube_array2 = np.broadcast_to(array2, (dim,) * 3)

    # Transform the second cube array for proper element alignment
    cube_array2 = np.moveaxis(cube_array2, 2, 0)

    # fullprint(array1, 'array1.txt')

    # Sum the cubes
    cube = cube_array1 + cube_array2

    # Calculate the maximum values for axis=0
    max_array = np.amax(cube, axis=0)

    # Calculate the indices of the maximum values for axis=0
    max_indicies = np.argmax(cube, axis=0)

    # Return the maximum values and indicies of those values as two 2D arrays
    return (max_array, max_indicies)


def best_header(header_probs, bag=None):
    if bag is None:
        bag = wb.WordBag()

    label_count = len(bag.get_labels())
    header_array = np.fromiter(
        map(lambda x: x['probability'], bag.header_probabilities()), np.float64
    ).reshape(-1, label_count)

    kdtree = spspatial.KDTree(header_array)
    _, nearest = kdtree.query(header_probs, 1)
    return nearest


def all_equal(*items):
    return all(items[0] == item for item in items)


def test1():
    TEST_STRING = '75 WEST COMMERCIAL STREET SUITE 104'

    words = TEST_STRING.upper().split()

    bag = wb.WordBag()
    word_probs_list = word_probabilities(bag, *words)

    labels = [x['name'] for x in bag.get_labels()]
    print(TEST_STRING)
    for word, word_probs in zip(words, word_probs_list):
        print('{: >14} : {}'.format(word, labels[word_probs.argmax()]))
        all_probs = sorted(zip(labels, word_probs), key=lambda x: x[1], reverse=True)
        for i, (name, prob) in enumerate(all_probs):
            if i >= 3:
                break
            print('{}{: <15} = {}'.format(' ' * 40, name, prob))


def test2():
    FILE_PATH = 'labeled_files/e34563c.json'
    tag_manager = tm.TagManager.from_json(FILE_PATH)
    classify_tag_manager(tag_manager)
    with open('labeled_files/e34563c_headers.json', 'w') as save_file:
        tag_manager.write_json(save_file)


def test3():
    FILE_PATH = 'labeled_files/e34563c.json'
    tag_manager = tm.TagManager.from_json(FILE_PATH)
    header_indicies = classify_tag_manager(tag_manager)

    for index in tag_manager.index_iterator():
        cell = tag_manager.get(*index)
        if cell is None:
            continue
        # Otherwise cell is a tm.TagCell instance
        header_int = header_indicies[cell.column]
        cell_inputs = NetworkInput.inputs_from_tag_cell(cell, header_int)

        print(cell_inputs)
        return  # Only do one for now


if __name__ == '__main__':
    test2()
