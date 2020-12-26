import csv

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

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
	return tsv_name, data


def tsv_to_dataset(dataset_path, tsv, tokenizer, min_duration=None, max_duration=None):
	tsv_name = tsv[0]
	tsv_data = tsv[1]
	bitrate = 16000
	waveforms = []  # encoder_inputs
	labels = []
	for row in tsv_data:
		filename = row[0]
		label = preprocess_label(row[1])
		waveform = file_to_waveform(dataset_path + "clips/" + filename)
		waveforms.append(waveform)
		labels.append(label)

	print("{0} Total waveforms {1}".format(tsv_name, len(waveforms)))
	if min_duration:
		filtered_labels = []
		filtered_waveforms = []

		for label, waveform in zip(labels, waveforms):
			if waveform.shape[0] / bitrate > min_duration:
				filtered_waveforms.append(waveform)
				filtered_labels.append(label)
		print("{0} Removed {1} waveform with duration less than {2}".format(tsv_name, len(waveforms) - len(filtered_waveforms), min_duration))
		waveforms = filtered_waveforms
		labels = filtered_labels

	if max_duration:
		filtered_labels = []
		filtered_waveforms = []

		for label, waveform in zip(labels, waveforms):
			if waveform.shape[0] / bitrate < max_duration:
				filtered_waveforms.append(waveform)
				filtered_labels.append(label)

		print("{0} Removed {1} waveform with duration more than {2}".format(tsv_name, len(waveforms) - len(filtered_waveforms), max_duration))
		waveforms = filtered_waveforms
		labels = filtered_labels

	print("{0} Loaded waveforms {1}".format(tsv_name, len(waveforms)))

	encoder_input = pad_sequences(waveforms, padding='post', dtype="float32", value=0.0)

	tokenizer.fit_on_texts(labels)
	decoder_input = tokenizer.texts_to_sequences(labels)

	decoder_input = pad_sequences(decoder_input, padding='post', value=tokenizer.word_index[" "])
	decoder_target = np.concatenate((decoder_input[:, 1:], np.full((decoder_input.shape[0], 1), tokenizer.word_index[" "])), axis=1)

	ds = tf.data.Dataset.from_tensor_slices(((encoder_input, decoder_input), decoder_target))

	return ds
