from scipy.io import wavfile

from .text_preprocessing import unicode_to_ascii_from_texts, add_space_between_word_punctuation, create_vocab


def preprocess_cv(tsv, clips_path, character_level, to_lower, to_ascii, min_duration=None, max_duration=None, bitrate=16000):
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
