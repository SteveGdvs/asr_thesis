import re
import unicodedata

import numpy as np

from .constants import WORD_START_TOKEN, WORD_END_TOKEN


def unicode_to_ascii(unicode_str):
	ascii_str = ""
	unicode_str = unicodedata.normalize('NFD', unicode_str)

	for c in unicode_str:
		if unicodedata.category(c) != 'Mn':  # remove accents etc.
			ascii_str = ascii_str + c
	return ascii_str


def unicode_to_ascii_from_texts(input_texts, target_texts):
	processed_in_txt = []
	processed_tar_txt = []

	for in_txt, tar_txt in zip(input_texts, target_texts):
		processed_in_txt.append(unicode_to_ascii(in_txt))
		processed_tar_txt.append(unicode_to_ascii(tar_txt))

	return processed_in_txt, processed_tar_txt


def remove_numbers_from_texts(input_texts, target_texts):
	processed_in_txt = []
	processed_tar_txt = []

	numbers_match = re.compile(r"(\d+)(:?)")
	for in_txt, tar_txt in zip(input_texts, target_texts):
		in_tmp = numbers_match.sub(" ", in_txt)
		tar_tmp = numbers_match.sub(" ", tar_txt)
		in_tmp = " ".join(in_tmp.split())  # remove extra whitespaces
		tar_tmp = " ".join(tar_tmp.split())  # remove extra whitespaces
		processed_in_txt.append(in_tmp)
		processed_tar_txt.append(tar_tmp)

	return processed_in_txt, processed_tar_txt


def add_space_between_word_punctuation(input_texts, target_texts):
	processed_in_txt = []
	processed_tar_txt = []

	numbers_match = re.compile(r"([?.!,;:'])")
	for in_txt, tar_txt in zip(input_texts, target_texts):
		in_tmp = numbers_match.sub(r" \1 ", in_txt)
		tar_tmp = numbers_match.sub(r" \1 ", tar_txt)
		in_tmp = " ".join(in_tmp.split())  # remove extra whitespaces
		tar_tmp = " ".join(tar_tmp.split())  # remove extra whitespaces
		processed_in_txt.append(in_tmp)
		processed_tar_txt.append(tar_tmp)

	return processed_in_txt, processed_tar_txt


def reverse_one_hot_characters(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 3:
		sequences = [sequences]

	for sentence in sequences:
		out = ""
		for char in sentence:
			out = out + tokenizer.index_word[np.argmax(char)]
		result.append(out.strip())

	return result


def reverse_one_hot_words(sequences, tokenizer):
	result = []

	if len(sequences.shape) != 3:
		sequences = [sequences]

	for sentence in sequences:
		out = []
		for word in sentence:
			decoded_word = tokenizer.index_word[np.argmax(word)]
			if decoded_word != WORD_START_TOKEN and decoded_word != WORD_END_TOKEN:
				out.append(decoded_word)
		result.append(" ".join(out))

	return result
