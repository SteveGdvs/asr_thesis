from .constants import CHARACTER_START_TOKEN, CHARACTER_END_TOKEN, WORD_START_TOKEN, WORD_END_TOKEN, WORD_PAD_TOKEN, CHARACTER_PAD_TOKEN
from .text_preprocessing import remove_numbers_from_texts, unicode_to_ascii_from_texts, add_space_between_word_punctuation, create_vocab


def preprocess_text_pairs_for_seq2seq(text_pairs, character_level, to_lower, to_ascii, remove_numbers):
	input_texts = [pair[0] for pair in text_pairs]
	target_texts = [pair[1] for pair in text_pairs]

	if remove_numbers:
		input_texts = remove_numbers_from_texts(input_texts)
		target_texts = remove_numbers_from_texts(target_texts)

	if to_ascii:
		input_texts = unicode_to_ascii_from_texts(input_texts)
		target_texts = unicode_to_ascii_from_texts(target_texts)

	if to_lower:
		input_texts = [text.lower() for text in input_texts]
		target_texts = [text.lower() for text in target_texts]

	if character_level:
		target_texts = [CHARACTER_START_TOKEN + sentence + CHARACTER_END_TOKEN for sentence in target_texts]
		input_vocab = create_vocab(input_texts+[CHARACTER_PAD_TOKEN], character_level)
		target_vocab = create_vocab(target_texts+[CHARACTER_PAD_TOKEN], character_level)

	else:
		input_texts = add_space_between_word_punctuation(input_texts)
		target_texts = add_space_between_word_punctuation(target_texts)

		target_texts = [WORD_START_TOKEN + " " + sentence + " " + WORD_END_TOKEN for sentence in target_texts]

		input_vocab = create_vocab(input_texts+[WORD_PAD_TOKEN], character_level)
		target_vocab = create_vocab(target_texts+[WORD_PAD_TOKEN], character_level)

	return (input_texts, target_texts), (input_vocab, target_vocab)
