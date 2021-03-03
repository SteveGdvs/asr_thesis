from typing import List

from tensorflow import keras

from .ctc_layer import CTCLayer


def create_ctc_model(rnn_type, rnn_dims: List[int], input_dim: int, output_dim: int, cnn_dims: List[int] = None, optimizer=None, line_length=150):
	rnn_type = rnn_type.lower()
	if rnn_type == "lstm":
		bidirectional = False
		RnnLayer = keras.layers.LSTM
	elif rnn_type == "bi_lstm":
		bidirectional = True
		RnnLayer = keras.layers.LSTM
	elif rnn_type == "gru":
		bidirectional = False
		RnnLayer = keras.layers.GRU
	elif rnn_type == "bi_gru":
		bidirectional = True
		RnnLayer = keras.layers.GRU
	else:
		raise ValueError("rnn_type must be one of the following: {0}".format(["lstm", "bi_lstm", "gru", "bi_gru"]))

	if optimizer is None:
		print("Using default optimizer RMSprop()")
		optimizer = keras.optimizers.RMSprop()

	name = rnn_type + "_model_" + "_".join([str(dim) for dim in rnn_dims])

	# input layers
	features_input = keras.layers.Input(shape=(None, input_dim), name="features_input")
	labels_input = keras.layers.Input(name="labels_input", shape=(None,))
	input_length = keras.layers.Input(name='features_len', shape=[1], dtype='int32')
	label_length = keras.layers.Input(name='labels_len', shape=[1], dtype='int32')
	x = features_input

	if cnn_dims:
		for i, dim in enumerate(cnn_dims):
			x = keras.layers.Conv1D(dim, 3, name="cnn_{0}".format(i))(x)

	if bidirectional:
		for i, dim in enumerate(rnn_dims):
			x = keras.layers.Bidirectional(RnnLayer(dim, return_sequences=True), name="rnn_{0}".format(i))(x)
	else:
		for i, dim in enumerate(rnn_dims):
			x = RnnLayer(dim, return_sequences=True, name="rnn_{0}".format(i))(x)

	# output and ctc loss
	x = keras.layers.Dense(output_dim + 1, activation="softmax", name="dense")(x)
	output = CTCLayer(blank_index=output_dim, name="ctc_loss")(labels_input, x, input_length, label_length)

	model = keras.models.Model(inputs=[features_input, labels_input, input_length, label_length], outputs=output, name=name)

	model.compile(optimizer=optimizer)

	model.summary(line_length)
	return model


def get_prediction_model(model, line_length=150):
	prediction_model = keras.models.Model(model.get_layer(name="features_input").input, model.get_layer(name="dense").output, name=model.name + "_prediction")
	prediction_model.summary(line_length)
	return prediction_model
