from abc import ABC, abstractmethod


class Seq2Seq(ABC):

	@abstractmethod
	def create_model(self, optimizer, line_length, summary):
		pass

	@abstractmethod
	def fit(self, data, epochs, validation_data, callbacks, **kwargs):
		pass

	@abstractmethod
	def decode_sequence(self, input_seq):
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
