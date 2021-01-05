from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from tensorflow import keras


class Seq2Seq(ABC):

	@abstractmethod
	def create_model(self, optimizer, line_length, summary):
		pass

	@abstractmethod
	def fit(self, data, epochs, validation_data, callbacks, **kwargs):
		pass

	@abstractmethod
	def evaluate(self, data, **kwargs):
		pass

	@abstractmethod
	def decode_sequences(self, input_seq):
		pass

	@classmethod
	@abstractmethod
	def load(cls, location):
		pass

	@abstractmethod
	def save(self, location):
		pass

	@abstractmethod
	def plot_history(self, figsize):
		pass

	@abstractmethod
	def get_history(self):
		pass

	@property
	@abstractmethod
	def name(self):
		pass

	@staticmethod
	def get_default_optimizer():
		print("Using default optimizer: RMSprop()")
		return keras.optimizers.RMSprop()

	@staticmethod
	def _plot_history(history, name, figsize):
		fig, axs = plt.subplots(2, figsize=figsize)
		fig.suptitle(name)

		axs[0].set_title("Train and Validation loss")
		axs[0].plot(history["loss"], label="train loss")
		axs[0].plot(history["val_loss"], label="validation loss")

		axs[1].set_title("Train and Validation accuracy")
		axs[1].plot(history["accuracy"], label="train accuracy")
		axs[1].plot(history["val_accuracy"], label="validation accuracy")

		axs[0].set_xlabel('Loss')
		axs[1].set_xlabel('Accuracy')

		axs[1].set_xlabel('Epochs')
		axs[0].legend()
		axs[1].legend()

		fig.subplots_adjust(hspace=0.3)
		fig.show()
