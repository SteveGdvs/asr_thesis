import re
import unicodedata


def unicode_to_ascii(unicode_str):
	ascii_str = ""
	unicode_str = unicodedata.normalize('NFD', unicode_str)

	for c in unicode_str:
		if unicodedata.category(c) != 'Mn':  # remove accents etc.
			ascii_str = ascii_str + c
	return ascii_str


def unicode_to_ascii_from_texts(texts):
	processed_txt = []

	for txt in texts:
		processed_txt.append(unicode_to_ascii(txt))

	return processed_txt


def remove_numbers_from_texts(texts):
	processed_txt = []

	numbers_match = re.compile(r"(\d+)(:?)")
	for txt in texts:
		tmp = numbers_match.sub(" ", txt)
		tmp = " ".join(tmp.split())  # remove extra whitespaces
		processed_txt.append(tmp)

	return processed_txt


def add_space_between_word_punctuation(texts):
	processed_txt = []

	numbers_match = re.compile(r"([?.!,;:'])")
	for txt in texts:
		tmp = numbers_match.sub(r" \1 ", txt)
		tmp = " ".join(tmp.split())  # remove extra whitespaces
		processed_txt.append(tmp)

	return processed_txt


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
