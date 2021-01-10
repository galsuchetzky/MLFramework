"""
This file contains the models used in the project.

Usage:
-   Implement here the architecture of your project.
-   There is a simple model architecture provided below, you may use it to implement your model.
-   If there are custom layers that are shared between your models, you may implement these in the Layers.py file to
    maintain readability.

Author:
    Gal Suchetzky (galsuchetzky@gmail.com)
"""

import Layers
import torch
import torch.nn as nn


# TODO: add your models here, you can use the following implementation template:
"""
class <model name>(nn.Module):
    def __init__(self, <initialization params>):
        super(<model name>, self).__init__()
        # TODO: Define the layers of your model here:

    def forward(self, <forward input>):
        # TODO: Add forward pass logic and save the result in the out parameter.
        return out
"""


class NerBiLstmModel(torch.nn.Module):
	"""
	Implements a BiLSTM network with an embedding layer and
	single hidden layer.
	This network will predict a sequence of labels (e.g. PER) for a
	given token (e.g. Henry) using a featurized window around the token.
	"""

	def __init__(self, helper, config, pretrained_embeddings):
		"""
		- Initialize the layer of the models:
			- *Unfrozen* embeddings of shape (V, D) which are loaded with pre-trained weights (`pretrained_embeddings`)
			- BiLSTM layer with hidden size of H/2 per direction
			- Linear layer with output of shape C

			Where:
			V - size of the vocabulary
			D - size of a word embedding
			H - size of the hidden layer
			C - number of classes being predicted

		Hints:
		- For the input dimension of the BiLSTM, think about the size of an embedded word representation
		"""
		super(NerBiLstmModel, self).__init__()
		self.config = config
		self._max_length = min(config.max_length, helper.max_length)

		self._dropout = torch.nn.Dropout(config.dropout)

		### YOUR CODE HERE (3 lines)
		# expect input of shape (batch_size, max_length, n_features)
		# and embed each one separately to get output of (batch_size, max_length, n_features, config.embed_size)
		self._embedding = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
		# change LSTM's input_size in order to fit the entire embedding of a word in one dim (instead of 2)
		self._bilstm = torch.nn.LSTM(input_size=config.embed_size * config.n_features,
		                             hidden_size=config.hidden_size // 2,
		                             batch_first=True, num_layers=1, bidirectional=True)
		self._output_layer = torch.nn.Linear(in_features=config.hidden_size, out_features=config.n_classes)
		# output dim : (batch_size, max_length, n_classes)
		# softmax dim is 3 because we want rescaling over classes
		self._softmax = torch.nn.LogSoftmax(dim=2)

	### END YOUR CODE

	def forward(self, sentences):
		"""
		- Perform the forward pass of the model, according to the model description in the handout:
			1. Get the embeddings of the input
			2. Apply dropout on the output of 1
			3. Pass the output of 2 through the BiLSTM layer
			4. Apply dropout on the output of 3
			5. Pass the output of 4 through the linear layer
			6. Perform softmax on the output of 5 to get tag_probs

		Hints:
		- Reshape the output of the embeddings layer so the full representation of an embedded word fits in one dimension.
		  You might find the .view method of a tensor helpful.

		Args:
		- sentences: The input tensor of shape (batch_size, max_length, n_features)

		Returns:
		- tag_probs: A tensor of shape (batch_size, max_length, n_classes) which represents the probability
					 for each tag for each word in a sentence.
		"""
		batch_size, seq_length = sentences.shape[0], sentences.shape[1]
		### YOUR CODE HERE (5-9 lines)
		embedded_input = self._dropout(self._embedding(sentences.long())).view(batch_size, seq_length, -1)
		output, (h_n, c_n) = self._bilstm(embedded_input)
		output = self._dropout(output)
		output = self._output_layer(output)
		tag_probs = self._softmax(output)
		### END YOUR CODE
		return tag_probs