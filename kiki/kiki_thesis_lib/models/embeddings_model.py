import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from kiki_thesis_lib.models.abstract_model import Seq2Seq
from kiki_thesis_lib.utils import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, WORD_START_TOKEN, WORD_END_TOKEN


class EmbeddingSeq2Seq(Seq2Seq):

	def __init__(self, rnn_type, rnn_size, embedding_size, num_encoder_tokens, num_decoder_tokens, target_token_index, target_index_token, max_decoder_seq_length, character_level):
		acceptable_rnn_types = ("lstm", "gru", "bi_lstm", "bi_gru")

		if rnn_type is None or rnn_type.lower() not in acceptable_rnn_types:
			raise ValueError("rnn_type must be one of the following: {0}".format(acceptable_rnn_types))

		self._rnn_type = rnn_type.lower()
		self._rnn_size = rnn_size
		self._embedding_size = embedding_size

		self._train_history = None
		self._model = None
		self._inference_models = None

		self._character_level = character_level
		self._num_decoder_tokens = num_decoder_tokens
		self._num_encoder_tokens = num_encoder_tokens
		self._max_decoder_seq_length = max_decoder_seq_length
		self._target_token_index = target_token_index
		self._target_index_token = target_index_token

	def create_model(self, optimizer=None, line_length=300, summary=True):

		if optimizer is None:
			optimizer = keras.optimizers.RMSprop(learning_rate=0.01)

		if self._rnn_type == "lstm":
			self._model = self._create_lstm()
		elif self._rnn_type == "gru":
			self._model = self._create_gru()
		elif self._rnn_type == "bi_lstm":
			self._model = self._create_bidirectional_lstm()
		elif self._rnn_type == "bi_gru":
			self._model = self._create_bidirectional_gru()

		self._model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

		if summary:
			self._model.summary(line_length)

	def _create_lstm(self):
		# encoder input
		encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
		# encoder embedding
		encoder_embedding = keras.layers.Embedding(self._num_encoder_tokens, self._embedding_size, name="encoder_embedding")(encoder_inputs)
		# encoder
		encoder = keras.layers.LSTM(self._rnn_size, return_state=True, name="encoder")
		encoder_outputs, state_h, state_c = encoder(encoder_embedding)
		encoder_states = [state_h, state_c]
		# decoder input
		decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
		# decoder embedding
		decoder_embedding = keras.layers.Embedding(self._num_decoder_tokens, self._embedding_size, name="decoder_embedding")(decoder_inputs)
		# decoder
		decoder = keras.layers.LSTM(self._rnn_size, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name="EmbeddingSeq2Seq_{0}".format(self._rnn_type))
		return model

	def _create_gru(self):
		# encoder input
		encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
		# encoder embedding
		encoder_embedding = keras.layers.Embedding(self._num_encoder_tokens, self._embedding_size, name="encoder_embedding")(encoder_inputs)
		# encoder
		encoder = keras.layers.GRU(self._rnn_size, return_state=True, name="encoder")
		encoder_outputs, state = encoder(encoder_embedding)
		encoder_states = [state]
		# decoder input
		decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
		# decoder embedding
		decoder_embedding = keras.layers.Embedding(self._num_decoder_tokens, self._embedding_size, name="decoder_embedding")(decoder_inputs)
		# decoder
		decoder = keras.layers.GRU(self._rnn_size, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _ = decoder(decoder_embedding, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name="EmbeddingSeq2Seq_{0}".format(self._rnn_type))
		return model

	def _create_bidirectional_lstm(self):
		# encoder input
		encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
		# encoder embedding
		encoder_embedding = keras.layers.Embedding(self._num_encoder_tokens, self._embedding_size, name="encoder_embedding")(encoder_inputs)
		# encoder
		encoder = keras.layers.Bidirectional(keras.layers.LSTM(self._rnn_size, return_state=True), name="encoder")
		encoder_outputs, fstate_h, fstate_c, bstate_h, bstate_c = encoder(encoder_embedding)
		state_h = keras.layers.Concatenate()([fstate_h, bstate_h])
		state_c = keras.layers.Concatenate()([fstate_c, bstate_c])
		encoder_states = [state_h, state_c]
		# decoder input
		decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
		# decoder embedding
		decoder_embedding = keras.layers.Embedding(self._num_decoder_tokens, self._embedding_size, name="decoder_embedding")(decoder_inputs)
		# decoder
		decoder = keras.layers.LSTM(self._rnn_size * 2, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name="EmbeddingSeq2Seq_{0}".format(self._rnn_type))
		return model

	def _create_bidirectional_gru(self):
		# encoder input
		encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
		# encoder embedding
		encoder_embedding = keras.layers.Embedding(self._num_encoder_tokens, self._embedding_size, name="encoder_embedding")(encoder_inputs)
		# encoder
		encoder = keras.layers.Bidirectional(keras.layers.GRU(self._rnn_size, return_state=True), name="encoder")
		encoder_outputs, fstate, bstate = encoder(encoder_embedding)
		state = keras.layers.Concatenate()([fstate, bstate])
		encoder_states = [state]
		# decoder input
		decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
		# decoder embedding
		decoder_embedding = keras.layers.Embedding(self._num_decoder_tokens, self._embedding_size, name="decoder_embedding")(decoder_inputs)
		# decoder
		decoder = keras.layers.GRU(self._rnn_size * 2, return_sequences=True, return_state=True, name="decoder")
		decoder_outputs, _ = decoder(decoder_embedding, initial_state=encoder_states)
		# decoder dense output
		decoder_dense = keras.layers.Dense(self._num_decoder_tokens, activation='softmax', name="decoder_dense")
		dense_outputs = decoder_dense(decoder_outputs)

		model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs, name="EmbeddingSeq2Seq_{0}".format(self._rnn_type))
		return model

	def fit(self, data, epochs, validation_data=None, callbacks=None, **kwargs):
		if self._model is not None:
			history = self._model.fit(data, epochs=epochs, validation_data=validation_data, callbacks=callbacks, **kwargs)
			self._train_history = history.history

		else:
			raise ValueError("You must first create a model before calling fit")

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
		dec_embedding = self._model.get_layer("decoder_embedding")(decoder_inputs)
		if self._rnn_type == "lstm" or self._rnn_type == "bi_lstm":
			decoder_state_input_h = keras.Input(shape=(latent_dim,))
			decoder_state_input_c = keras.Input(shape=(latent_dim,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		else:
			decoder_state_input = keras.Input(shape=(latent_dim,))
			decoder_states_inputs = [decoder_state_input]
		decoder_lstm = self._model.get_layer("decoder")
		decoder_output_and_states = decoder_lstm(dec_embedding, initial_state=decoder_states_inputs)
		decoder_outputs = decoder_output_and_states[0]
		decoder_states = decoder_output_and_states[1:]
		decoder_dense = self._model.get_layer("decoder_dense")
		decoder_outputs = decoder_dense(decoder_outputs)
		decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

		return encoder_model, decoder_model

	def decode_sequence(self, input_seq):
		target_token_index = self._target_token_index
		reverse_target_token_index = self._target_index_token

		if self._character_level:
			start_token = CHARACTER_START_TOKEN
			end_token = CHARACTER_END_TOKEN
			join_str = ""
		else:
			start_token = WORD_START_TOKEN
			end_token = WORD_END_TOKEN
			join_str = " "

		if self._inference_models is None:
			encoder_model, decoder_model = self._create_inference_models()
			self._inference_models = encoder_model, decoder_model
		else:
			encoder_model, decoder_model = self._inference_models

		states_value = encoder_model.predict(input_seq)

		if self._rnn_type == "gru" or self._rnn_type == "bi_gru":
			states_value = [states_value]

		# Set the first character of target sequence with the start token.
		target_seq = np.array([[target_token_index[start_token]]])

		# Sampling loop for a batch of sequences
		# (to simplify, here we assume a batch of size 1).
		stop_condition = False
		decoded_sentence = []

		while not stop_condition:
			output_tokens_and_states = decoder_model.predict([target_seq] + states_value)

			output_tokens = output_tokens_and_states[0]
			states_value = output_tokens_and_states[1:]

			# Sample a token
			sampled_token_index = np.argmax(output_tokens[0, -1, :])
			sampled_token = reverse_target_token_index[sampled_token_index]

			# Exit condition: either hit max length
			# or find stop character.
			if sampled_token == end_token or len(decoded_sentence) > self._max_decoder_seq_length:
				stop_condition = True
			else:
				decoded_sentence.append(sampled_token)

			# Update the target sequence (of length 1).
			target_seq = np.array([[sampled_token_index]])

		return join_str.join(decoded_sentence)

	@classmethod
	def load(cls, location):
		model = keras.models.load_model(location + "/model")
		with open(location + '/attributes.pickle', 'rb') as handle:
			attributes = pickle.load(handle)
		new_cls = cls(
			rnn_type=attributes["rnn_type"],
			rnn_size=attributes["rnn_size"],
			embedding_size=attributes["embedding_size"],
			num_encoder_tokens=attributes["num_encoder_tokens"],
			num_decoder_tokens=attributes["num_decoder_tokens"],
			character_level=attributes["character_level"],
			max_decoder_seq_length=attributes["max_decoder_seq_length"],
			target_token_index=attributes["target_token_index"],
			target_index_token=attributes["target_index_token"]
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
				"embedding_size": self._embedding_size,
				"num_encoder_tokens": self._num_encoder_tokens,
				"num_decoder_tokens": self._num_decoder_tokens,
				"character_level": self._character_level,
				"train_history": self._train_history,
				"target_token_index": self._target_token_index,
				"target_index_token": self._target_index_token
			}
			with open(location + '/attributes.pickle', 'wb') as handle:
				pickle.dump(attributes, handle)
		else:
			raise ValueError("You must first create a model before saving it")

	def plot_history(self, figsize=(25, 15)):
		fig, axs = plt.subplots(2, figsize=figsize)
		if self._character_level:
			fig.suptitle("EmbeddingSeq2Seq: {0} Character level tokenization".format(self._rnn_type))
		else:
			fig.suptitle("EmbeddingSeq2Seq {0} Word level tokenization".format(self._rnn_type))
		axs[0].set_title("Train loss and accuracy")
		axs[0].plot(self._train_history["loss"], label="train loss")
		axs[0].plot(self._train_history["val_loss"], label="validation loss")

		axs[1].set_title("Validation loss and accuracy")
		axs[1].plot(self._train_history["accuracy"], label="train accuracy")
		axs[1].plot(self._train_history["val_accuracy"], label="Validation accuracy")

		axs[0].set_xlabel('Loss')
		axs[1].set_xlabel('Accuracy')

		axs[1].set_xlabel('Epochs')
		axs[0].legend()
		axs[1].legend()

		fig.subplots_adjust(hspace=0.3)
		fig.show()

	def get_history(self):
		return self._train_history