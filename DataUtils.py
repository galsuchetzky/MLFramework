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