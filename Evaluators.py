"""
Predictor template:
class Predictor:
	def __init__(self, model, config):
		self._model = model
		self._config = config

	def predict(self, examples, use_str_labels=False):
		config = self._config
		preprocessed_examples = ???

		preds = []
		prog = Progbar(target=1 + int(len(preprocessed_examples) / config.batch_size))
		for i, minibatch in enumerate(minibatches(preprocessed_examples, config.batch_size, shuffle=False)):
			# TODO add predictions on the minibatch to the preds list
			prog.update(i + 1, [])

		return preds
"""

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


"""
Evaluator template:
class Evaluator:
	def __init__(self, predictor):
		self._predictor = predictor

	def evaluate(self, examples):
		Here calculate the scores according to any matric you choose and return the scores.

"""
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