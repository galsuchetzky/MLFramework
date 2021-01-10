# TODO add here trainer base and trainer template
class TrainerBase:
	def __init__(self, model, config, helper, logger):
		self._model = model
		self._config = config
		self._helper = helper
		self._logger = logger

		self._evaluator = Evaluator(Predictor(model, config))

	def train(self, train_examples, dev_examples):
		model = self._model
		config = self._config
		logger = self._logger

		best_score = 0.

		preprocessed_train_examples = train_examples['preprocessed']
		step = 0
		for epoch in range(config.n_epochs):
			model.train()
			logger.info("Epoch %d out of %d", epoch + 1, config.n_epochs)
			prog = Progbar(target=1 + int(len(preprocessed_train_examples) / config.batch_size))

			avg_loss = 0
			for i, minibatch in enumerate(minibatches(preprocessed_train_examples, config.batch_size)):
				sentences = torch.tensor(minibatch[0], device=config.device)
				labels = torch.tensor(minibatch[1], device=config.device)
				masks = torch.tensor(minibatch[2], device=config.device)
				avg_loss += self._train_on_batch(sentences, labels, masks)
			avg_loss /= i + 1
			logger.info("Training average loss: %.5f", avg_loss)

			model.eval()
			with torch.no_grad():
				logger.info("Evaluating on development data")
				token_cm, entity_scores = self._evaluator.evaluate(dev_examples)
				logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
				logger.debug("Token-level scores:\n" + token_cm.summary())
				logger.info("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*entity_scores))

				score = entity_scores[-1]

				if score > best_score and config.model_output:
					best_score = score
					logger.info("New best score! Saving model in %s", config.model_output)
					torch.save(model.state_dict(), config.model_output)
				print("")
		return best_score

	def _train_on_batch(self, sentences, labels, masks):
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

		### YOUR CODE HERE (1 line)
		# because later the tag_probs are ***after the softmax***
		self._loss_function = torch.nn.NLLLoss()
		### END YOUR CODE
		self._optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

	def _train_on_batch(self, sentences, labels, masks):
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
		### YOUR CODE HERE (3-6 lines)
		masked_labels = (labels * masks).long()
		padded_masks = masks.unsqueeze(-1).expand(tag_probs.size()).float()
		masked_tag_probs = tag_probs * padded_masks
		masked_tag_probs = torch.transpose(masked_tag_probs, dim0=1, dim1=2)
		### END YOUR CODE
		loss = self._loss_function(masked_tag_probs, masked_labels)
		return loss


class Predictor:
	def __init__(self, model, config):
		self._model = model
		self._config = config

	def predict(self, examples, use_str_labels=False):
		"""
		Reports the output of the model on examples (uses helper to featurize each example).
		"""
		config = self._config
		preprocessed_examples = examples['preprocessed']

		preds = []
		prog = Progbar(target=1 + int(len(preprocessed_examples) / config.batch_size))
		for i, minibatch in enumerate(minibatches(preprocessed_examples, config.batch_size, shuffle=False)):
			sentences = torch.tensor(minibatch[0], device=config.device)
			tag_probs = self._model(sentences)
			preds_ = torch.argmax(tag_probs, dim=-1)
			preds += list(preds_)
			prog.update(i + 1, [])

		return self.consolidate_predictions(examples, preds, use_str_labels)

	@staticmethod
	def consolidate_predictions(examples, preds, use_str_labels=False):
		"""Batch the predictions into groups of sentence length.
		"""
		assert len(examples['tokens']) == len(examples['preprocessed'])
		assert len(examples['tokens']) == len(preds)

		ret = []
		for i, (sentence, labels) in enumerate(examples['token_indices'] if not use_str_labels else examples['tokens']):
			_, _, mask = examples['preprocessed'][i]
			labels_ = [l.item() for l, m in zip(preds[i], mask) if m]  # only select elements of mask.
			assert len(labels_) == len(labels)
			ret.append([sentence, labels, labels_])
		return ret


class Evaluator:
	def __init__(self, predictor):
		self._predictor = predictor

	def evaluate(self, examples):
		"""Evaluates model performance on @examples.

		This function uses the model to predict labels for @examples and constructs a confusion matrix.

		Returns:
			The F1 score for predicting tokens as named entities.
		"""
		token_cm = ConfusionMatrix(labels=LBLS)

		correct_preds, total_correct, total_preds = 0., 0., 0.
		for data in self._predictor.predict(examples):
			(_, labels, labels_) = data

			for l, l_ in zip(labels, labels_):
				token_cm.update(l, l_)
			gold = set(get_chunks(labels))
			pred = set(get_chunks(labels_))
			correct_preds += len(gold.intersection(pred))
			total_preds += len(pred)
			total_correct += len(gold)

		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correct if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
		return token_cm, (p, r, f1)