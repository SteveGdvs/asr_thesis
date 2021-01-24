import numpy as np
import tensorflow as tf

from thesis_lib.preprocessing.constants import WORD_START_TOKEN, WORD_END_TOKEN, WORD_PAD_TOKEN


def reverse_one_hot(sequences, num_to_vocab, character_level):
	if character_level:
		separator = ""
	else:
		separator = " "

	result = []
	for r in sequences:
		sentence = tf.strings.reduce_join(num_to_vocab(tf.argmax(r, axis=1)), separator=separator).numpy().decode("utf-8")
		if not character_level:
			sentence = sentence.replace(WORD_START_TOKEN, "")
			sentence = sentence.replace(WORD_END_TOKEN, "")
			sentence = sentence.replace(WORD_PAD_TOKEN, "")
		sentence = sentence.strip()
		result.append(sentence)

	if len(result) == 1:
		result = result[0]
	return result


def reverse_tokenization(sequences, num_to_vocab, character_level):
	if character_level:
		separator = ""
	else:
		separator = " "

	result = []
	for r in sequences:
		sentence = tf.strings.reduce_join(num_to_vocab(r), separator=separator).numpy().decode("utf-8")
		if not character_level:
			sentence = sentence.replace(WORD_START_TOKEN, "")
			sentence = sentence.replace(WORD_END_TOKEN, "")
			sentence = sentence.replace(WORD_PAD_TOKEN, "")
		sentence = sentence.strip()
		result.append(sentence)

	if len(result) == 1:
		result = result[0]
	return result


def decode_ctc_batch_predictions(preds, num_to_vocab, pred_lens, character_level):
	if character_level:
		separator = ""
	else:
		separator = " "

	(decoded,), _ = tf.nn.ctc_greedy_decoder(inputs=np.moveaxis(preds, [0, 1, 2], [1, 0, 2]), sequence_length=pred_lens)
	results = tf.sparse.to_dense(decoded, default_value=-1)

	# Iterate over the results and get back the text
	output_text = []
	for res in results:
		res = res[res > 0]
		res = tf.strings.reduce_join(num_to_vocab(res), separator=separator).numpy().decode("utf-8")
		output_text.append(res.strip())
	return output_text


def get_string_lookup(vocab):
	vocab_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab), num_oov_indices=0, mask_token=None)
	num_to_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocab_to_num.get_vocabulary(), invert=True, mask_token=None)

	return vocab_to_num, num_to_vocab
