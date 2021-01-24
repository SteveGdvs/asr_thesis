import wave

from tqdm import tqdm

from .constants import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, WORD_END_TOKEN, WORD_START_TOKEN, CHARACTER_PAD_TOKEN, WORD_PAD_TOKEN
from .text_preprocessing import unicode_to_ascii_from_texts, add_space_between_word_punctuation, create_vocab


def preprocess_cv_s2s(tsv, clips_path, character_level, to_lower, to_ascii):
	tsv = [(clips_path + pair[0], pair[1].strip()) for pair in tsv]

	print("Loaded waveforms {0}".format(len(tsv)))

	labels = [pair[1] for pair in tsv]
	audio_files = [pair[0] for pair in tsv]

	if to_ascii:
		labels = unicode_to_ascii_from_texts(labels)
	if to_lower:
		labels = [label.lower() for label in labels]
	if not character_level:
		labels = add_space_between_word_punctuation(labels)

	if character_level:
		labels = [CHARACTER_START_TOKEN + label + CHARACTER_END_TOKEN for label in labels]

		vocab = create_vocab(labels + [CHARACTER_PAD_TOKEN], character_level)

	else:
		labels = add_space_between_word_punctuation(labels)
		labels = [WORD_START_TOKEN + " " + label + " " + WORD_END_TOKEN for label in labels]

		vocab = create_vocab(labels + [WORD_PAD_TOKEN], character_level)

	maxlen = len(max(labels, key=len))

	return list(zip(audio_files, labels)), vocab, maxlen


def preprocess_cv_ctc(tsv, clips_path, character_level, to_lower, to_ascii):
	print("Loaded waveforms {0}".format(len(tsv)))

	audio_files = [clips_path + pair[0] for pair in tsv]
	labels = [pair[1] for pair in tsv]

	if to_ascii:
		labels = unicode_to_ascii_from_texts(labels)
	if to_lower:
		labels = [label.lower() for label in labels]
	if not character_level:
		labels = add_space_between_word_punctuation(labels)

	vocab = create_vocab(labels, character_level)
	maxlen = len(max(labels, key=len))

	return list(zip(audio_files, labels)), vocab, maxlen


def filter_cv(tsv, clips_path, min_duration=None, max_duration=None, bitrate=16000):
	if min_duration is not None or max_duration is not None:
		new_tsv = []
		min_duration = min_duration or 0
		max_duration = max_duration or 100

		for audio_file, label in tqdm(tsv, desc="Checking wav duration"):
			with wave.open(clips_path + audio_file, 'rb') as wf:
				frames = wf.getnframes()
				duration = frames / bitrate
				if min_duration < duration < max_duration:
					new_tsv.append((audio_file, label))

		print("Removed {0} tsv entries with duration less than {1} and bigger than {2}".format(len(tsv) - len(new_tsv), min_duration, max_duration))
		tsv = new_tsv
		print("Final tsv entries {0}".format(len(tsv)))
	return tsv
