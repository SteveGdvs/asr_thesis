import tensorflow as tf
import tensorflow_io as tfio


def file_to_waveform(filename):
	audio_binary = tf.io.read_file(filename)
	waveform, _ = tf.audio.decode_wav(audio_binary)
	return tf.squeeze(waveform, axis=-1)


def preprocess_label(label):
	return "\t" + label.strip() + "\n"


def to_spectrogram(input_data, label_sequence):
	audio = input_data[0]
	audio = tf.cast(audio, tf.float32)
	audio = audio / 32768.0
	spectrogram = tfio.experimental.audio.spectrogram(audio, nfft=256, window=512, stride=128)
	return (spectrogram, input_data[1]), label_sequence


def to_mel_spectrogram(input_data, label_sequence):
	spectrogram = input_data[0]
	mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
	return (mel_spectrogram, input_data[1]), label_sequence


def to_dbscale_mel_spectrogram(input_data, label_sequence):
	mel_spectrogram = input_data[0]
	dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
	return (dbscale_mel_spectrogram, input_data[1]), label_sequence


def to_one_hot_sequence(input_data, label_sequence, vocab_size):
	decoder_input_data = tf.one_hot(input_data[1], depth=vocab_size)
	decoder_target_data = tf.one_hot(label_sequence, depth=vocab_size)
	return (input_data[0], decoder_input_data), decoder_target_data
