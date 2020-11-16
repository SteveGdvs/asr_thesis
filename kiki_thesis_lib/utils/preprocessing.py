import re
import unicodedata

import numpy as np

from .constants import WORD_START_TOKEN, WORD_END_TOKEN, WORD_PAD_TOKEN


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


def reverse_one_hot_character_level(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 3:
		sequences = [sequences]

	for sentence in sequences:
		out = ""
		for char in sentence:
			out = out + tokenizer.index_word[np.argmax(char)]
		result.append(out.strip())

	return result


def reverse_one_hot_word_level(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 3:
		sequences = [sequences]

	for sentence in sequences:
		out = []
		for word in sentence:
			decoded_word = tokenizer.index_word[np.argmax(word)]
			if decoded_word != WORD_START_TOKEN and decoded_word != WORD_END_TOKEN and decoded_word != WORD_PAD_TOKEN:
				out.append(decoded_word)
		result.append(" ".join(out))

	return result


def reverse_tokenization_character_level(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 2:
		sequences = [sequences]

	for sentence in sequences:
		out = ""
		for char in sentence:
			out = out + tokenizer.index_word[char]
		result.append(out.strip())

	return result


def reverse_tokenization_word_level(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 2:
		sequences = [sequences]

	for sentence in sequences:
		out = []
		for word in sentence:
			decoded_word = tokenizer.index_word[word]
			if decoded_word != WORD_START_TOKEN and decoded_word != WORD_END_TOKEN and decoded_word != WORD_PAD_TOKEN:
				out.append(decoded_word)
		result.append(" ".join(out))

	return result
