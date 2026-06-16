import numpy as np

from regularization.l2 import L2
from visualization.decision_boundary import plot_decision_boundary

from datasets.circles import Circles
from datasets.moons import Moons
from layers.model import Model


from metrics.accuracy import Accuracy
from metrics.precision import Precision
from metrics.recall import Recall
from metrics.f1 import F1_score

class Trainer:
	def __init__(self, model, loss_fn, optimizer, learning_rate=0.01):
		self.model = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.learning_rate = learning_rate

	def _regularization_loss(self):
		total_loss = 0.0

		for layer in getattr(self.model, "layers", []):
			regularizer = getattr(layer, "regularizer", None)

			if regularizer is not None:
				total_loss += regularizer.penalty(layer.weights)

		return total_loss

	def fit(
		self,
		X,
		y,
		epochs=1000,
		verbose=True,
		log_every=100,
	):
		history = {
			"loss": [],
			"accuracy": [],
			"precision": [],
			"recall": [],
			"f1": [],
		}

		for epoch in range(epochs):
			pred = self.model.forward(X)

			loss = self.loss_fn.forward(y, pred)
			grad = self.loss_fn.backward(y, pred)

			self.model.backward(
				grad,
				self.learning_rate,
				self.optimizer,
			)

			total_loss = loss + self._regularization_loss()
			accuracy = Accuracy()(y, pred)
			precision = Precision()(y, pred)
			recall = Recall()(y, pred)
			f1 = F1_score()(y, pred)

			history["loss"].append(total_loss)
			history["accuracy"].append(accuracy)
			history["precision"].append(precision)
			history["recall"].append(recall)
			history["f1"].append(f1)

			if verbose and (
				epoch == 0
				or (epoch + 1) % log_every == 0
				or epoch == epochs - 1
			):
				print(
					f"Epoch {epoch + 1}/{epochs} "
					f"Loss {total_loss:.4f} "
					f"Accuracy {accuracy:.2%} "
					f"Precision {precision:.2%} "
					f"Recall {recall:.2%} "
					f"F1 {f1:.2%}"
				)

		return history

	def predict(self, X, threshold=0.5):
		return (self.model.forward(X) > threshold).astype(int)



	def evaluate(self, X, y):
		pred = self.model.forward(X)

		loss = self.loss_fn.forward(y, pred) + self._regularization_loss()
		accuracy = Accuracy()(y, pred)
		precision = Precision()(y, pred)
		recall = Recall()(y, pred)
		f1 = F1_score()(y, pred)

		return {
			"loss": loss,
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}


def train(
	model,
	X,
	y,
	loss_fn,
	optimizer,
	epochs=1000,
	learning_rate=0.01,
	verbose=True,
	log_every=100,
):
	trainer = Trainer(
		model=model,
		loss_fn=loss_fn,
		optimizer=optimizer,
		learning_rate=learning_rate,
	)

	return trainer.fit(
		X=X,
		y=y,
		epochs=epochs,
		verbose=verbose,
		log_every=log_every,
	)

