"""
This file defines the training loops of your models as Trainers.

Usage:
-   See the main function in Train.py.

Code adapted from TAU-NLP course exercise code.

Author:
    Gal Suchetzky (galsuchetzky@gmail.com)
    Itay Levy (itaylevy4@gmail.com)

TODOs:
-
"""
import torch

from Utils import Progbar, minibatches
from Evaluators import Evaluator, Predictor

# TODO add here trainer base and trainer template
class TrainerBase:
	def __init__(self, model, config, helper, logger):
		self._model = model
		self._config = config
		# self._helper = helper
		self._logger = logger

		# For evaluating the model predictions.
		self._evaluator = Evaluator(Predictor(model, config))

	def train(self, train_examples, dev_examples):
		model = self._model
		config = self._config
		logger = self._logger

		# For keeping track of the best score while training.
		best_score = 0.

		# Get examples.
		# TODO if no processing of the train examples is needed, remove this line.
		preprocessed_train_examples = train_examples['preprocessed']

		step = 0
		for epoch in range(config.n_epochs):
			# Set model on train mode.
			model.train()

			logger.info("Epoch %d out of %d", epoch + 1, config.n_epochs)
			Progbar(target=1 + int(len(preprocessed_train_examples) / config.batch_size))

			avg_loss = 0
			# Break the training examples to minibatches and train on each minibatch.
			for i, minibatch in enumerate(minibatches(preprocessed_train_examples, config.batch_size)):
				batch_args = self._get_batch_args(minibatch)
				avg_loss += self._train_on_batch(batch_args)
			avg_loss /= i + 1
			logger.info("Training average loss: %.5f", avg_loss)

			# Set the model to evaluate mode and evaluate it.
			model.eval()
			with torch.no_grad():
				logger.info("Evaluating on development data")
				token_cm, entity_scores = self._evaluator.evaluate(dev_examples)
				# logger.debug("Token-level scores:\n" + token_cm.summary())
				# logger.info("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*entity_scores))

				# Save the model with the best score.
				score = entity_scores[-1]
				if score > best_score and config.model_output:
					best_score = score
					logger.info("New best score! Saving model in %s", config.model_output)
					torch.save(model.state_dict(), config.model_output)
				print("")
		return best_score

	def _train_on_batch(self, batch_args):
		raise NotImplementedError

	def _get_batch_args(self, minibatch):
		raise NotImplementedError


class Trainer(TrainerBase):
	def __init__(self, model, config, helper, logger):
		"""
		- Define the cross entropy loss function in self._loss_function.
		  It will be used in _batch_loss.

		Hints:
		- Don't use automatically PyTorch's CrossEntropyLoss - read its documentation first
		"""
		super(Trainer, self).__init__(model, config, helper, logger)

		self._loss_function = torch.nn.NLLLoss()
		self._optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

		# scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

	def _get_batch_args(self, minibatch):
		config = self._config
		sentences = torch.tensor(minibatch[0], device=config.device)
		labels = torch.tensor(minibatch[1], device=config.device)
		masks = torch.tensor(minibatch[2], device=config.device)
		batch_args = (sentences, labels, masks)
		return batch_args

	def _train_on_batch(self, batch_args):
		sentences, labels, masks = batch_args
		model = self._model
		config = self._config

		tag_probs = model(sentences)

		model.zero_grad()
		batch_loss = self._batch_loss(tag_probs, labels, masks)
		batch_loss.backward()

		self._optimizer.step()

		return batch_loss

	def _batch_loss(self, tag_probs, labels, masks):
		"""
		- Calculate the cross entropy loss of the input batch

		Hints:
		- You might find torch.unsqueeze, torch.masked_fill (use ~masks to get the inversion) and torch.transpose useful.

		Args:
		- tag_probs: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
					 network (*after* softmax).
		- labels: The gold labels tensor of shape (batch_size, max_length)
		- masks: The masks tensor of shape (batch_size, max_length)

		Returns:
			loss: A 0-d tensor (scalar)
		"""
		masked_labels = (labels * masks).long()
		padded_masks = masks.unsqueeze(-1).expand(tag_probs.size()).float()
		masked_tag_probs = tag_probs * padded_masks
		masked_tag_probs = torch.transpose(masked_tag_probs, dim0=1, dim1=2)
		loss = self._loss_function(masked_tag_probs, masked_labels)
		return loss



