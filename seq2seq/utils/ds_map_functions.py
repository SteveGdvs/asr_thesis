import tensorflow as tf
import tensorflow_io as tfio


def to_spectrogram(input_data, label_sequence):
	spectrogram = tfio.experimental.audio.spectrogram(input_data[0], nfft=2048, window=2048, stride=512)  # nfft=512, window=512, stride=128
	return (spectrogram, input_data[1]), label_sequence


def to_mel_spectrogram_from_spect(input_data, label_sequence):
	spectrogram = input_data[0]
	mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)  # mels = 64 if nfft=256 or verify graph
	return (mel_spectrogram, input_data[1]), label_sequence


def to_dbscale_mel_spectrogram_from_mel(input_data, label_sequence):
	mel_spectrogram = input_data[0]
	dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
	return (dbscale_mel_spectrogram, input_data[1]), label_sequence


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
