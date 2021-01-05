import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from scipy.io import wavfile

from seq2seq.utils import unicode_to_ascii_from_texts, add_space_between_word_punctuation, file_to_waveform


def preprocess_for_ctc_model(tsv, clips_path, character_level, to_lower, to_ascii, min_duration=None, max_duration=None, bitrate=16000):
	tsv = [(clips_path + pair[0], pair[1].strip()) for pair in tsv]

	print("Total waveforms {0}".format(len(tsv)))
	if min_duration:
		new_tsv = []
		for audio_file_path, label in tsv:
			_, waveform = wavfile.read(audio_file_path)
			if waveform.shape[0] / bitrate > min_duration:
				new_tsv.append((audio_file_path, label))
		print("Removed {0} waveform with duration less than {1}".format(len(tsv) - len(new_tsv), min_duration))
		tsv = new_tsv

	if max_duration:
		new_tsv = []
		for audio_file_path, label in tsv:
			_, waveform = wavfile.read(audio_file_path)
			if waveform.shape[0] / bitrate < max_duration:
				new_tsv.append((audio_file_path, label))

		print("Removed {0} waveform with duration more than {1}".format(len(tsv) - len(new_tsv), max_duration))
		tsv = new_tsv

	print("Loaded waveforms {0}".format(len(tsv)))

	labels = [pair[1] for pair in tsv]
	audio_files = [pair[0] for pair in tsv]

	if to_ascii:
		labels = unicode_to_ascii_from_texts(labels)
	if to_lower:
		labels = [label.lower() for label in labels]
	if not character_level:
		labels = add_space_between_word_punctuation(labels)

	vocab = create_vocab(labels, character_level)

	maxlen = len(max(labels, key=len))

	return list(zip(audio_files, labels)), vocab, maxlen


def decode_batch_predictions(pred, num_to_vocab, max_length, character_level):
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


def to_mfccs_ctc(input_data, n_mfccs, vocab_to_num, character_level):
	waveform = file_to_waveform(input_data[0])
	spectrogram = tfio.experimental.audio.spectrogram(waveform, nfft=2048, window=2048, stride=512)
	mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :n_mfccs]
	mfccs = mfccs - tf.math.reduce_mean(mfccs) / tf.math.reduce_std(mfccs)
	mfccs = mfccs[..., :n_mfccs]
	if character_level:
		label_seq = vocab_to_num(tf.strings.unicode_split(input_data[1], input_encoding="UTF-8"))
	else:
		label_seq = vocab_to_num(tf.strings.split(input_data[1]))

	return {"features_input": mfccs, "labels_input": label_seq}


def create_vocab(texts, character_level):
	vocab = []

	if character_level:
		for text in texts:
			for char in text:
				vocab.append(char)
	else:
		for text in texts:
			vocab.extend(text.split())

	return set(vocab)
