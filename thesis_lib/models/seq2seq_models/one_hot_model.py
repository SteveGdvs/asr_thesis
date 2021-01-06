import pickle

import numpy as np
from tensorflow import keras

from .abstract_model import Seq2Seq
from ...preprocessing.constants import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, WORD_START_TOKEN, WORD_END_TOKEN, CHARACTER_PAD_TOKEN, WORD_PAD_TOKEN
from ...preprocessing.utils import reverse_tokenization


class OneHotSeq2Seq(Seq2Seq):

	def __init__(self, rnn_type, rnn_size, num_encoder_tokens, num_decoder_tokens, target_token_to_index, target_index_to_token, max_decoder_seq_length, character_level):
		acceptable_rnn_types = ("lstm", "gru", "bi_lstm", "bi_gru")

		if rnn_type is None or rnn_type.lower() not in acceptable_rnn_types:
			raise ValueError("rnn_type must be one of the following: {0}".format(acceptable_rnn_types))

		self._rnn_type = rnn_type.lower()
		self._rnn_size = rnn_size
		self._num_decoder_tokens = num_decoder_tokens
		self._num_encoder_tokens = num_encoder_tokens
		self._character_level = character_level
		self._max_decoder_seq_length = max_decoder_seq_length
		self._target_token_to_index = target_token_to_index
		self._target_index_to_token = target_index_to_token

		self._train_history = None
		self._model = None
		self._inference_models = None

	@property
	def name(self):
		if self._character_level:
			return "OneHotSeq2Seq_{0}_{1}_character_level".format(self._rnn_type, self._rnn_size)
		else:
			return "OneHotSeq2Seq_{0}_{1}_word_level".format(self._rnn_type, self._rnn_size)

	def create_model(self, optimizer=None, metrics=None, line_length=300, summary=True):

		if optimizer is None:
			optimizer = super(OneHotSeq2Seq, self).get_default_optimizer()

		if metrics is None:
			metrics = ["accuracy"]
		else:
			if "accuracy" not in metrics:
				metrics.insert(0, "accuracy")

		if self._rnn_type == "lstm":
			self._model = self._create_lstm()
		elif self._rnn_type == "gru":
			self._model = self._create_gru()
		elif self._rnn_type == "bi_lstm":
			self._model = self._create_bidirectional_lstm()
		elif self._rnn_type == "bi_gru":
			self._model = self._create_bidirectional_gru()

		self._model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(), metrics=metrics)

		if summary:
			self._model.summary(line_length)

	def _create_lstm(self):
		# encoder input
		encoder_inputs = keras.Input(shape=(None, self._num_encoder_tokens), name="encoder_input")
		# encoder
		encoder = keras.layers.LSTM(self._rnn_size, return_state=True, name="encoder")
		_, state_h, state_c = encoder(encoder_inputs)
		encoder_states = [state_h, state_c]
		# decoder input
		decoder_inputs = keras.Input(shape=(None, self._num_decoder_tokens), name="decoder_input")
		# decoder
		decoder = keras.layers.LSTM(self._rnn_size, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name=self.name)
		return model

	def _create_gru(self):
		# encoder input
		encoder_inputs = keras.Input(shape=(None, self._num_encoder_tokens), name="encoder_input")
		# encoder
		encoder = keras.layers.GRU(self._rnn_size, return_state=True, name="encoder")
		_, state = encoder(encoder_inputs)
		encoder_states = [state]
		# decoder input
		decoder_inputs = keras.Input(shape=(None, self._num_decoder_tokens), name="decoder_input")
		# decoder
		decoder = keras.layers.GRU(self._rnn_size, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name=self.name)
		return model

	def _create_bidirectional_lstm(self):
		# encoder input
		encoder_inputs = keras.Input(shape=(None, self._num_encoder_tokens), name="encoder_input")
		# encoder
		encoder = keras.layers.Bidirectional(keras.layers.LSTM(self._rnn_size, return_state=True), name="encoder")
		_, fstate_h, fstate_c, bstate_h, bstate_c = encoder(encoder_inputs)
		state_h = keras.layers.Concatenate()([fstate_h, bstate_h])
		state_c = keras.layers.Concatenate()([fstate_c, bstate_c])
		encoder_states = [state_h, state_c]
		# decoder input
		decoder_inputs = keras.Input(shape=(None, self._num_decoder_tokens), name="decoder_input")
		# decoder
		decoder = keras.layers.LSTM(self._rnn_size * 2, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name=self.name)
		return model

	def _create_bidirectional_gru(self):
		# encoder input
		encoder_inputs = keras.Input(shape=(None, self._num_encoder_tokens), name="encoder_input")
		# encoder
		encoder = keras.layers.Bidirectional(keras.layers.GRU(self._rnn_size, return_state=True), name="encoder")
		_, fstate, bstate = encoder(encoder_inputs)
		state = keras.layers.Concatenate()([fstate, bstate])
		encoder_states = [state]
		# decoder input
		decoder_inputs = keras.Input(shape=(None, self._num_decoder_tokens), name="decoder_input")
		# decoder
		decoder = keras.layers.GRU(self._rnn_size * 2, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name=self.name)
		return model

	def fit(self, data, epochs, validation_data=None, callbacks=None, **kwargs):
		if self._model is not None:
			history = self._model.fit(data, epochs=epochs, validation_data=validation_data, callbacks=callbacks, **kwargs)
			self._train_history = history.history

		else:
			raise ValueError("You must first create a model before calling fit")

	def evaluate(self, data, **kwargs):
		return self._model.evaluate(data)

	def _create_inference_models(self):
		encoder_inputs = self._model.get_layer("encoder_input").input  # input_1
		encoder_output_and_states = self._model.get_layer("encoder").output  # lstm_1

		if self._rnn_type == "bi_lstm":
			state_h = keras.layers.Concatenate()([encoder_output_and_states[1], encoder_output_and_states[3]])
			state_c = keras.layers.Concatenate()([encoder_output_and_states[2], encoder_output_and_states[4]])
			encoder_states = [state_h, state_c]
			latent_dim = self._rnn_size * 2
		elif self._rnn_type == "bi_gru":
			state = keras.layers.Concatenate()([encoder_output_and_states[1], encoder_output_and_states[2]])
			encoder_states = [state]
			latent_dim = self._rnn_size * 2
		else:
			encoder_states = encoder_output_and_states[1:]
			latent_dim = self._rnn_size

		encoder_model = keras.Model(encoder_inputs, encoder_states)

		decoder_inputs = self._model.get_layer("decoder_input").input  # input_2
		if self._rnn_type == "lstm" or self._rnn_type == "bi_lstm":
			decoder_state_input_h = keras.Input(shape=(latent_dim,))
			decoder_state_input_c = keras.Input(shape=(latent_dim,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		else:
			decoder_state_input = keras.Input(shape=(latent_dim,))
			decoder_states_inputs = [decoder_state_input]
		decoder_lstm = self._model.get_layer("decoder")
		decoder_output_and_states = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
		decoder_outputs = decoder_output_and_states[0]
		decoder_states = decoder_output_and_states[1:]
		decoder_dense = self._model.get_layer("decoder_dense")
		decoder_outputs = decoder_dense(decoder_outputs)
		decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

		return encoder_model, decoder_model

	def _trim_sos_eos(self, sentences_indexes):

		if self._character_level:
			end_token_index = self._target_token_to_index(CHARACTER_END_TOKEN).numpy()
			pad_token = self._target_token_to_index(CHARACTER_PAD_TOKEN).numpy()
		else:
			end_token_index = self._target_token_to_index(WORD_END_TOKEN).numpy()
			pad_token = self._target_token_to_index(WORD_PAD_TOKEN).numpy()

		new_sentences_indexes = []
		for sentence in sentences_indexes:
			for i, values in enumerate(sentence):
				if values == end_token_index:
					sentence[i:] = pad_token
					break
			new_sentences_indexes.append(sentence)
		return new_sentences_indexes

	def decode_sequences(self, input_seq):
		target_token_to_index = self._target_token_to_index
		reverse_target_token_index = self._target_index_to_token

		if self._character_level:
			start_token_index = target_token_to_index(CHARACTER_START_TOKEN).numpy()
			end_token_index = target_token_to_index(CHARACTER_END_TOKEN).numpy()
		else:
			start_token_index = target_token_to_index(WORD_START_TOKEN).numpy()
			end_token_index = target_token_to_index(WORD_END_TOKEN).numpy()

		if self._inference_models is None:
			encoder_model, decoder_model = self._create_inference_models()
			self._inference_models = encoder_model, decoder_model
		else:
			encoder_model, decoder_model = self._inference_models

		states_value = encoder_model.predict(input_seq)

		if self._rnn_type == "gru" or self._rnn_type == "bi_gru":
			states_value = [states_value]

		# Generate empty target sequence of length 1.
		target_seq = np.zeros((input_seq.shape[0], 1, self._num_decoder_tokens))
		# Populate the first character of target sequence with the start character.
		target_seq[:, 0, start_token_index] = 1.0

		# Sampling loop for a batch of sequences
		# (to simplify, here we assume a batch of size 1).
		stop_condition = False
		decoded_sentences_indexes = []

		while not stop_condition:
			output_tokens_and_states = decoder_model.predict([target_seq] + states_value)

			output_tokens = output_tokens_and_states[0]
			states_value = output_tokens_and_states[1:]

			# Sample a token
			sampled_token_index = np.argmax(output_tokens, axis=2)

			# Exit condition: either hit max length
			# or find stop character.
			# if sampled_token_index == end_token_index or len(decoded_sentence_indexes) > self._max_decoder_seq_length:
			if np.all(sampled_token_index.flatten() == 0) or np.all(sampled_token_index.flatten() == end_token_index) or len(decoded_sentences_indexes) > self._max_decoder_seq_length:
				stop_condition = True
			else:
				decoded_sentences_indexes.append(sampled_token_index)

			# Update the target sequence (of length 1).
			target_seq = np.zeros((input_seq.shape[0], 1, self._num_decoder_tokens))
			for i, idx in enumerate(sampled_token_index):
				target_seq[i, 0, idx] = 1.0

		decoded_sentences_indexes = np.stack(np.squeeze(decoded_sentences_indexes), axis=1)
		decoded_sentences_indexes = self._trim_sos_eos(decoded_sentences_indexes)
		return reverse_tokenization(decoded_sentences_indexes, reverse_target_token_index, self._character_level)

	@classmethod
	def load(cls, location):
		model = keras.models.load_model(location + "/model")
		with open(location + '/attributes.pickle', 'rb') as handle:
			attributes = pickle.load(handle)
		new_cls = cls(
			rnn_type=attributes["rnn_type"],
			rnn_size=attributes["rnn_size"],
			num_encoder_tokens=attributes["num_encoder_tokens"],
			num_decoder_tokens=attributes["num_decoder_tokens"],
			character_level=attributes["character_level"],
			max_decoder_seq_length=attributes["max_decoder_seq_length"],
			target_token_to_index=attributes["target_token_to_index"],
			target_index_to_token=attributes["target_index_to_token"]
		)
		new_cls._model = model
		new_cls._train_history = attributes["train_history"]
		return new_cls

	def save(self, location):
		if self._model is not None:
			self._model.save(location + "/model")

			attributes = {
				"rnn_type": self._rnn_type,
				"rnn_size": self._rnn_size,
				"num_encoder_tokens": self._num_encoder_tokens,
				"num_decoder_tokens": self._num_decoder_tokens,
				"character_level": self._character_level,
				"max_decoder_seq_length": self._max_decoder_seq_length,
				"target_token_to_index": self._target_token_to_index,
				"target_index_to_token": self._target_index_to_token,
				"train_history": self._train_history,
			}
			with open(location + '/attributes.pickle', 'wb') as handle:
				pickle.dump(attributes, handle)
		else:
			raise ValueError("You must first create a model before saving it")

	def plot_history(self, figsize=(25, 15)):
		super(OneHotSeq2Seq, self)._plot_history(history=self._train_history, name=self.name, figsize=figsize)

	def get_history(self):
		return self._train_history
