import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .constants import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, CHARACTER_PAD_TOKEN, WORD_END_TOKEN, WORD_START_TOKEN, WORD_PAD_TOKEN
from .preprocessing import unicode_to_ascii, remove_numbers_from_texts, unicode_to_ascii_from_texts, add_space_between_word_punctuation, reverse_one_hot_characters, reverse_one_hot_words


def load_data(path, lines_limit=None, max_length=None, remove_duplicates=True, remove_in_list=None):
	with open(path, encoding="utf_8") as f:
		lines = f.readlines()

	if lines_limit is not None:
		lines = lines[: min(lines_limit, len(lines) - 1)]

	input_texts = []
	target_texts = []
	for line in lines:
		parts = line.split("\t")
		input_text = parts[0].strip()
		target_text = parts[1].strip()
		input_texts.append(input_text)
		target_texts.append(target_text)

	print("Read {0} lines from \"{1}\"".format(len(lines), path))
	del lines  # free memory

	if max_length is not None:
		input_texts_max_length = []
		target_texts_max_length = []

		for in_txt, tar_txt in zip(input_texts, target_texts):
			if len(in_txt) < max_length and len(tar_txt) < max_length:
				input_texts_max_length.append(in_txt)
				target_texts_max_length.append(tar_txt)

		print("- Removed {0} sentences exceeding {1} characters".format(len(input_texts) - len(input_texts_max_length), max_length))
		input_texts = input_texts_max_length
		target_texts = target_texts_max_length
		del input_texts_max_length  # free memory
		del target_texts_max_length  # free memory

	if remove_duplicates:
		input_texts_no_dup = set()  # for speed
		target_texts_no_dup = []

		for in_txt, tar_txt in zip(input_texts, target_texts):
			if in_txt not in input_texts_no_dup:
				input_texts_no_dup.add(in_txt)
				target_texts_no_dup.append(tar_txt)

		print("- Removed {0} duplicates".format(len(input_texts) - len(input_texts_no_dup)))
		input_texts = list(input_texts_no_dup)
		target_texts = target_texts_no_dup
		del input_texts_no_dup  # free memory
		del target_texts_no_dup  # free memory

	if remove_in_list:
		for word in remove_in_list:
			input_texts_temp = []
			target_texts_temp = []
			for in_txt, tar_txt in zip(input_texts, target_texts):
				if word.lower() not in in_txt.lower():
					input_texts_temp.append(in_txt)
					target_texts_temp.append(tar_txt)

			print("- Removed {0} lines containing the word {1}".format((len(input_texts) - len(input_texts_temp)), word))
			input_texts = input_texts_temp
			target_texts = target_texts_temp

	if len(input_texts) != len(target_texts):
		raise ValueError("input_texts and target_texts must have the same length")

	print("Loaded {0} sentences".format(len(input_texts)))
	return input_texts, target_texts


def character_level_tokenization(input_texts, target_texts, to_lower=False, to_ascii=False, remove_numbers=False):
	if remove_numbers:
		input_texts, target_texts = remove_numbers_from_texts(input_texts, target_texts)

	if to_ascii:
		input_texts, target_texts = unicode_to_ascii_from_texts(input_texts, target_texts)

	target_texts = [CHARACTER_START_TOKEN + sentence + CHARACTER_END_TOKEN for sentence in target_texts]

	input_tokenizer = Tokenizer(filters='', char_level=True, lower=to_lower)
	input_tokenizer.fit_on_texts(input_texts)

	target_tokenizer = Tokenizer(filters='', char_level=True, lower=to_lower)
	target_tokenizer.fit_on_texts(target_texts)

	input_sequences = input_tokenizer.texts_to_sequences(input_texts)
	target_sequences = target_tokenizer.texts_to_sequences(target_texts)

	input_sequences = pad_sequences(input_sequences, padding='post', value=input_tokenizer.word_index[CHARACTER_PAD_TOKEN])
	target_sequences = pad_sequences(target_sequences, padding='post', value=target_tokenizer.word_index[CHARACTER_PAD_TOKEN])

	decoder_target_seq = np.concatenate((target_sequences[:, 1:], np.full((target_sequences.shape[0], 1), target_tokenizer.word_index[CHARACTER_PAD_TOKEN])), axis=1)

	return (input_sequences, target_sequences, decoder_target_seq), (input_tokenizer, target_tokenizer)


def word_level_tokenization(input_texts, target_texts, to_lower=True, to_ascii=True, remove_numbers=True):
	if remove_numbers:
		input_texts, target_texts = remove_numbers_from_texts(input_texts, target_texts)

	if to_ascii:
		input_texts, target_texts = unicode_to_ascii_from_texts(input_texts, target_texts)

	input_texts, target_texts = add_space_between_word_punctuation(input_texts, target_texts)

	input_texts = [WORD_START_TOKEN + " " + sentence + " " + WORD_END_TOKEN for sentence in input_texts]
	target_texts = [WORD_START_TOKEN + " " + sentence + " " + WORD_END_TOKEN for sentence in target_texts]

	input_tokenizer = Tokenizer(filters='', char_level=False, lower=to_lower)
	input_tokenizer.fit_on_texts(input_texts)

	target_tokenizer = Tokenizer(filters='', char_level=False, lower=to_lower)
	target_tokenizer.fit_on_texts(target_texts)

	input_sequences = input_tokenizer.texts_to_sequences(input_texts)
	target_sequences = target_tokenizer.texts_to_sequences(target_texts)

	input_sequences = pad_sequences(input_sequences, padding='post', value=input_tokenizer.word_index[WORD_PAD_TOKEN])
	target_sequences = pad_sequences(target_sequences, padding='post', value=target_tokenizer.word_index[WORD_PAD_TOKEN])

	decoder_target_seq = np.concatenate((target_sequences[:, 1:], np.full((target_sequences.shape[0], 1), target_tokenizer.word_index[WORD_PAD_TOKEN])), axis=1)

	return (input_sequences, target_sequences, decoder_target_seq), (input_tokenizer, target_tokenizer)


def prepare_for_one_hot_translate(sentences, input_tokenizer, max_encoder_seq_length, character_level):
	sequences = input_tokenizer.texts_to_sequences(sentences)
	if character_level:
		sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_encoder_seq_length, padding='post', value=input_tokenizer.word_index[CHARACTER_PAD_TOKEN])
	else:
		sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_encoder_seq_length, padding='post', value=input_tokenizer.word_index[WORD_PAD_TOKEN])

	sequences = tf.keras.utils.to_categorical(sequences, num_classes=len(input_tokenizer.index_word) + 1)

	return sequences


def one_hot_map(input_data, target_data, input_vocab_size, output_vocab_size):
	encoder_input_data = tf.one_hot(input_data[0], depth=input_vocab_size)
	decoder_input_data = tf.one_hot(input_data[1], depth=output_vocab_size)
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def one_hot_target_data_map(input_data, target_data, output_vocab_size):
	encoder_input_data = input_data[0]
	decoder_input_data = input_data[1]
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def reverse_one_hot(sequences, tokenizer, character_level):
	if character_level:
		return reverse_one_hot_characters(sequences, tokenizer)
	else:
		return reverse_one_hot_words(sequences, tokenizer)
