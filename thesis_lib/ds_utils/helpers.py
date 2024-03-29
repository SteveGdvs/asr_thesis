import pickle
from pathlib import Path

import tensorflow as tf


def file_to_waveform(filename):
	audio_binary = tf.io.read_file(filename)
	waveform, _ = tf.audio.decode_wav(audio_binary)
	return tf.squeeze(waveform, axis=-1)


def split_ds(ds: tf.data.Dataset, val_percentage=None, test_percentage=None, buffer_size=None):
	val_percentage = val_percentage or 0
	test_percentage = test_percentage or 0
	buffer_size = buffer_size or 128 * 128
	if val_percentage < 0 or val_percentage >= 1.0:
		raise ValueError("val_percentage must be between (0,1)")
	if test_percentage < 0 or test_percentage >= 1.0:
		raise ValueError("test_percentage must be between (0,1)")
	if (val_percentage + test_percentage) >= 1.0:
		raise ValueError("val_percentage+test_percentage must be between (0,1)")

	full_ds_size = len(ds)
	print("Full size: {0}".format(full_ds_size))
	if val_percentage == 0 and test_percentage == 0:
		print("No split returning ds shuffled")
		return ds.shuffle(buffer_size, reshuffle_each_iteration=False)
	elif val_percentage != 0 and test_percentage == 0:
		val_ds_size = int(full_ds_size * val_percentage)
		train_ds_size = full_ds_size - val_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		val_ds = ds.skip(train_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Val size: {0}".format(len(val_ds)))
		return train_ds, val_ds

	elif val_percentage == 0 and test_percentage != 0:
		test_ds_size = int(full_ds_size * test_percentage)
		train_ds_size = full_ds_size - test_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		test_ds = ds.skip(train_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Test size: {0}".format(len(test_ds)))
		return train_ds, test_ds
	else:
		val_ds_size = int(full_ds_size * val_percentage)
		test_ds_size = int(full_ds_size * test_percentage)
		train_ds_size = full_ds_size - test_ds_size - val_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		remaining = ds.skip(train_ds_size)

		test_ds = remaining.take(test_ds_size)
		val_ds = remaining.skip(test_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Val size: {0}".format(len(val_ds)))
		print("Test size: {0}".format(len(test_ds)))
		return train_ds, val_ds, test_ds,


def save_preprocessed_dataset(path: Path, dataset, vocab_to_num, num_to_vocab, character_level, vocab_size, max_sentence_length, name="saved_dataset"):
	path = path / name
	path.mkdir()

	info = dict()
	info["element_spec"] = dataset.element_spec
	info["vocab_to_num_weights"] = vocab_to_num.get_weights()
	info["num_to_vocab_weights"] = num_to_vocab.get_weights()
	info["character_level"] = character_level
	info["max_sentence_length"] = max_sentence_length
	info["vocab_size"] = vocab_size

	with open(path / 'info.pickle', 'wb') as handle:
		pickle.dump(info, handle)

	tf.data.experimental.save(dataset, str(path), compression="GZIP")

	print("Saved at {0}".format(path))


def load_preprocessed_dataset(path: Path):
	with open(path / 'info.pickle', 'rb') as handle:
		info = pickle.load(handle)

	element_spec = info["element_spec"]
	dataset = tf.data.experimental.load(str(path), element_spec=element_spec, compression="GZIP")

	vocab_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, mask_token=None)
	vocab_to_num.set_weights(info["vocab_to_num_weights"])
	num_to_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(invert=True, mask_token=None)
	num_to_vocab.set_weights(info["num_to_vocab_weights"])
	character_level = info["character_level"]
	max_sentence_length = info["max_sentence_length"]
	vocab_size = info["vocab_size"]

	return dataset, vocab_to_num, num_to_vocab, character_level, vocab_size, max_sentence_length
