import numpy as np
import tensorflow as tf

from thesis_lib.preprocessing.constants import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, CHARACTER_PAD_TOKEN, WORD_START_TOKEN, WORD_END_TOKEN, WORD_PAD_TOKEN


def reverse_one_hot(sequences, num_to_vocab, character_level):
	if character_level:
		separator = ""
		eos = CHARACTER_START_TOKEN
		sos = CHARACTER_END_TOKEN
		pad = CHARACTER_PAD_TOKEN
	else:
		separator = " "
		eos = WORD_START_TOKEN
		sos = WORD_END_TOKEN
		pad = WORD_PAD_TOKEN

	result = []

	for r in sequences:
		sentence = tf.strings.reduce_join(num_to_vocab(tf.argmax(r, axis=1)), separator=separator).numpy().decode("utf-8")
		sentence = sentence.replace(eos, "")
		sentence = sentence.replace(sos, "")
		sentence = sentence.replace(pad, "")
		sentence = sentence.strip()
		result.append(sentence)

	if len(result) == 1:
		result = result[0]
	return result


def reverse_tokenization(sequences, num_to_vocab, character_level):
	if character_level:
		separator = ""
		eos = CHARACTER_START_TOKEN
		sos = CHARACTER_END_TOKEN
		pad = CHARACTER_PAD_TOKEN
	else:
		separator = " "
		eos = WORD_START_TOKEN
		sos = WORD_END_TOKEN
		pad = WORD_PAD_TOKEN

	result = []
	for r in sequences:
		sentence = tf.strings.reduce_join(num_to_vocab(r), separator=separator).numpy().decode("utf-8")
		sentence = sentence.replace(eos, "")
		sentence = sentence.replace(sos, "")
		sentence = sentence.replace(pad, "")
		sentence = sentence.strip()
		result.append(sentence)

	if len(result) == 1:
		result = result[0]
	return result


def decode_ctc_batch_predictions(pred, num_to_vocab, max_length, character_level):
	if character_level:
		separator = ""
	else:
		separator = " "

	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	# Use greedy search. For complex tasks, you can use beam search
	results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

	results = tf.where(tf.math.equal(results, -1), 0, results)
	# Iterate over the results and get back the text
	output_text = []
	for res in results:
		res = tf.strings.reduce_join(num_to_vocab(res), separator=separator).numpy().decode("utf-8")
		output_text.append(res.strip())
	return output_text


def get_string_lookup(vocab):
	vocab_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab), num_oov_indices=0, mask_token=None)
	num_to_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocab_to_num.get_vocabulary(), invert=True, mask_token=None)

	return vocab_to_num, num_to_vocab
