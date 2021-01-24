import tensorflow as tf
import tensorflow_io as tfio

from .helpers import file_to_waveform


def _str_to_sequence(label, vocab_to_num, character_level):
	if character_level:
		label_seq = vocab_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
	else:
		label_seq = vocab_to_num(tf.strings.split(label))
	return tf.cast(label_seq, dtype=tf.int32), tf.shape(label_seq)[0]


def to_spectrogram(input_data, vocab_to_num, character_level):
	file = input_data[0]
	label = input_data[1]
	waveform = file_to_waveform(file)
	spectrogram = tfio.experimental.audio.spectrogram(waveform, nfft=2048, window=2048, stride=512)

	spectrogram_len = tf.shape(spectrogram)[0]

	label_seq, label_len = _str_to_sequence(label, vocab_to_num, character_level)

	return spectrogram, label_seq, spectrogram_len, label_len


def to_mel_spectrogram(input_data, vocab_to_num, character_level):
	spectrogram, label_seq, spectrogram_len, label_len = to_spectrogram(input_data, vocab_to_num, character_level)
	mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

	return mel_spectrogram, label_seq, spectrogram_len, label_len


def to_mfccs(input_data, n_mfccs, vocab_to_num, character_level):
	mel_spectrogram, label_seq, spectrogram_len, label_len = to_mel_spectrogram(input_data, vocab_to_num, character_level)
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :n_mfccs]
	mfccs = mfccs - tf.math.reduce_mean(mfccs) / tf.math.reduce_std(mfccs)
	mfccs = mfccs[..., :n_mfccs]
	return mfccs, label_seq, spectrogram_len, label_len


def to_one_hot_decoder_only(input_data, label_sequence, vocab_size):
	decoder_input_data = tf.one_hot(input_data[1], depth=vocab_size)
	decoder_target_data = tf.one_hot(label_sequence, depth=vocab_size)
	return (input_data[0], decoder_input_data), decoder_target_data


def to_one_hot_all(input_data, target_data, input_vocab_size, output_vocab_size):
	encoder_input_data = tf.one_hot(input_data[0], depth=input_vocab_size)
	decoder_input_data = tf.one_hot(input_data[1], depth=output_vocab_size)
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def to_one_hot_target_only(input_data, target_data, output_vocab_size):
	encoder_input_data = input_data[0]
	decoder_input_data = input_data[1]
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def to_tokenize_input_target(input_data, target_data, input_vocab_to_num, target_vocab_to_num, character_level):
	input_data_seq = _str_to_sequence(input_data, input_vocab_to_num, character_level)
	target_data_seq = _str_to_sequence(target_data, target_vocab_to_num, character_level)
	return input_data_seq, target_data_seq


def to_seq2seq_format(input_data, target_data, pad_value):
	return (input_data, target_data), tf.concat([target_data[1:], tf.constant([pad_value])], axis=0)


def to_ctc_format(input_data, target_data, input_len, target_len):
	return {"features_input": input_data, "labels_input": target_data, "features_len": input_len, "labels_len": target_len}
