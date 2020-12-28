import csv

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from kiki.seq2seq.utils import unicode_to_ascii_from_texts, add_space_between_word_punctuation, WORD_END_TOKEN, WORD_START_TOKEN, WORD_PAD_TOKEN, CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, \
	CHARACTER_PAD_TOKEN
from .dataset_helpers import preprocess_label, file_to_waveform


def read_tsv(dataset_path, tsv_name):
	data = []
	with open(dataset_path + tsv_name, encoding="utf-8") as train_tsc:
		reader = csv.reader(train_tsc, delimiter="\t", strict=True)
		next(reader)  # skip header
		for row in reader:
			file_name = row[1].replace(".mp3", ".wav")
			sentence = row[2]
			data.append((file_name, sentence))
	return data


def tsv_to_dataset(dataset_path, tsv_data, character_level, to_lower, to_ascii, min_duration=None, max_duration=None):
	bitrate = 16000
	waveforms = []  # encoder_inputs
	labels = []
	for row in tsv_data:
		filename = row[0]
		label = preprocess_label(row[1])
		waveform = file_to_waveform(dataset_path + "clips/" + filename)
		waveforms.append(waveform)
		labels.append(label)

	print("Total waveforms {0}".format(len(waveforms)))
	if min_duration:
		filtered_labels = []
		filtered_waveforms = []

		for label, waveform in zip(labels, waveforms):
			if waveform.shape[0] / bitrate > min_duration:
				filtered_waveforms.append(waveform)
				filtered_labels.append(label)
		print("Removed {0} waveform with duration less than {1}".format(len(waveforms) - len(filtered_waveforms), min_duration))
		waveforms = filtered_waveforms
		labels = filtered_labels

	if max_duration:
		filtered_labels = []
		filtered_waveforms = []

		for label, waveform in zip(labels, waveforms):
			if waveform.shape[0] / bitrate < max_duration:
				filtered_waveforms.append(waveform)
				filtered_labels.append(label)

		print("Removed {0} waveform with duration more than {1}".format(len(waveforms) - len(filtered_waveforms), max_duration))
		waveforms = filtered_waveforms
		labels = filtered_labels

	print("Loaded waveforms {0}".format(len(waveforms)))

	encoder_input = pad_sequences(waveforms, padding='post', dtype="float32", value=0.0)

	if to_ascii:
		labels = unicode_to_ascii_from_texts(labels)

	if character_level:
		labels = [CHARACTER_START_TOKEN + label + CHARACTER_END_TOKEN for label in labels]

		tokenizer = Tokenizer(filters='', char_level=character_level, lower=to_lower)
		tokenizer.fit_on_texts(labels)

		decoder_input = tokenizer.texts_to_sequences(labels)

		decoder_input = pad_sequences(decoder_input, padding='post', value=tokenizer.word_index[CHARACTER_PAD_TOKEN])
		decoder_target = np.concatenate((decoder_input[:, 1:], np.full((decoder_input.shape[0], 1), tokenizer.word_index[CHARACTER_PAD_TOKEN])), axis=1)

	else:
		labels = add_space_between_word_punctuation(labels)
		labels = [WORD_START_TOKEN + " " + label + " " + WORD_END_TOKEN for label in labels]

		tokenizer = Tokenizer(filters='', char_level=character_level, lower=to_lower)
		tokenizer.fit_on_texts(labels)
		tokenizer.fit_on_texts([WORD_PAD_TOKEN])

		decoder_input = tokenizer.texts_to_sequences(labels)

		decoder_input = pad_sequences(decoder_input, padding='post', value=tokenizer.word_index[WORD_PAD_TOKEN])
		decoder_target = np.concatenate((decoder_input[:, 1:], np.full((decoder_input.shape[0], 1), tokenizer.word_index[WORD_PAD_TOKEN])), axis=1)

	ds = tf.data.Dataset.from_tensor_slices(((encoder_input, decoder_input), decoder_target))
	max_sentence_length = len(max(labels, key=len))
	return ds, tokenizer, max_sentence_length
