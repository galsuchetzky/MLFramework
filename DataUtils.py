import torch.utils.data as data
import numpy as np

from Defaults import *


# TODO: add a dataset class here that manages the reading of the examples from the dataset.
#  you can use the following template:
class DatasetName(data.Dataset):
	"""
	<dataset description>
	Args:
		data_path (str): Path to .npz file containing pre-processed dataset.
	"""

	def __init__(self, data_path):
		super(DatasetName, self).__init__()

		self.dataset = np.load(data_path)
		# TODO add init code here

	def __getitem__(self, idx=-1):
		# TODO: retrieve a single example from the given idx location in the dataset.
		example = None

		return example

	def __len__(self):
		# TODO: define the  and return the len.
		pass

	def get_minibatch(self, minibatch_size=DEFAULT_MINIBATCH_SIZE):
		# TODO return a minibatch of examples
		pass

	def get_example(self, idx=-1):
		if idx == -1:
			idx = np.random.randint(len(self))
		return self.__getitem__(idx)


# TODO authors are these below necessary??

class BaseDataPreprocessor:
	def __init__(self, model, config, helper):
		self._max_length = model._max_length
		self._window_size = config.window_size
		self._n_features = config.n_features
		self._helper = helper

	def preprocess_sequence_data(self, examples):
		def featurize_windows(data, start, end, window_size=1):
			"""Uses the input sequences in @data to construct new windowed data points.
			"""
			ret = []
			for sentence, labels in data:
				from util import window_iterator
				sentence_ = []
				for window in window_iterator(sentence, window_size, beg=start, end=end):
					sentence_.append(sum(window, []))
				ret.append((sentence_, labels))
			return ret

		preprocessed_examples = featurize_windows(examples['token_indices'], self._helper.START, self._helper.END,
		                                          self._window_size)
		examples['preprocessed'] = self.pad_sequences(preprocessed_examples)
		return examples

	def pad_sequences(self, examples):
		raise NotImplementedError


class DataPreprocessor(BaseDataPreprocessor):
	def pad_sequences(self, examples):
		"""Ensures each input-output seqeunce pair in @data is of length
		@max_length by padding it with zeros and truncating the rest of the
		sequence.

		In the code below, for every sentence, labels pair in @data,
		(a) create a new sentence which appends zero feature vectors until
		the sentence is of length @max_length. If the sentence is longer
		than @max_length, simply truncate the sentence to be @max_length
		long.
		(b) create a new label sequence similarly.
		(c) create a _masking_ sequence that has a True wherever there was a
		token in the original sequence, and a False for every padded input.

		Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
		0, 0], and max_length = 5, we would construct
			- a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
			- a new label seqeunce: [1, 0, 0, 4, 4], and
			- a masking seqeunce: [True, True, True, False, False].

		Args:
			data: is a list of (sentence, labels) tuples. @sentence is a list
				containing the words in the sentence and @label is a list of
				output labels. Each word is itself a list of
				@n_features features. For example, the sentence "Chris
				Manning is amazing" and labels "PER PER O O" would become
				([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
				the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
				is the list of labels.
			max_length: the desired length for all input/output sequences.
		Returns:
			a new list of data points of the structure (sentence, labels, mask).
			Each of sentence, labels and mask are of length @max_length.
			See the example above for more details.
		"""
		ret = []

		max_length = self._max_length
		# Use this zero vector when padding sequences.
		zero_vector = [0] * self._n_features
		zero_label = 4  # corresponds to the 'O' tag

		for sentence, labels in examples:
			### YOUR CODE HERE (~5 lines)
			current_length = len(sentence)
			min_unmasked_length = min(current_length, max_length)
			zero_padding_length = max_length - min_unmasked_length

			# truncate for long sequences, else keep the same
			padded_sentence = sentence[:max_length]
			padded_labels = labels[:max_length]
			mask = [1] * min_unmasked_length

			# add padding
			padded_sentence += [zero_vector] * zero_padding_length
			padded_labels += [zero_label] * zero_padding_length
			mask += [0] * zero_padding_length

			ret.append((padded_sentence, padded_labels, mask))
		### END YOUR CODE
		return ret


def get_minibatches(data, minibatch_size, shuffle=True):
	"""
	Iterates through the provided data one minibatch at at time. You can use this function to
	iterate through data in minibatches as follows:

		for inputs_minibatch in get_minibatches(inputs, minibatch_size):
			...

	Or with multiple data sources:

		for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
			...

	Args:
		data: there are two possible values:
			- a list or numpy array
			- a list where each element is either a list or numpy array
		minibatch_size: the maximum number of items in a minibatch
		shuffle: whether to randomize the order of returned data
	Returns:
		minibatches: the return value depends on data:
			- If data is a list/array it yields the next minibatch of data.
			- If data a list of lists/arrays it returns the next minibatch of each element in the
			  list. This can be used to iterate through multiple data sources
			  (e.g., features and labels) at the same time.

	"""
	list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	data_size = len(data[0]) if list_data else len(data)
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)
	for minibatch_start in np.arange(0, data_size, minibatch_size):
		minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
		yield [minibatch(d, minibatch_indices) for d in data] if list_data \
			else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
	return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
	batches = [np.array(col) for col in zip(*data)]
	return get_minibatches(batches, batch_size, shuffle)
