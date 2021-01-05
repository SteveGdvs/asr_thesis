from typing import List

import jiwer
from nltk.translate import bleu_score
from nltk.translate import gleu_score

from .sentence import sentence_cer


def corpus_gleu(references: List[str], predictions: List[str]):
	if len(references) != len(predictions):
		raise ValueError("The lists must have the same length")

	references = [[o] for o in references]

	return gleu_score.corpus_gleu(references, predictions)


def corpus_bleu(references: List[str], predictions: List[str]):
	if len(references) != len(predictions):
		raise ValueError("The lists must have the same length")

	references = [[o] for o in references]

	return bleu_score.corpus_bleu(references, predictions)


def corpus_wer(references: List[str], predictions: List[str]):
	if len(references) != len(predictions):
		raise ValueError("The lists must have the same length")
	transformation = jiwer.Compose([
		jiwer.RemoveMultipleSpaces(),
		jiwer.RemovePunctuation(),
		jiwer.Strip(),
		jiwer.ToLowerCase(),
		jiwer.ExpandCommonEnglishContractions(),
		jiwer.RemoveWhiteSpace(replace_by_space=True),
		jiwer.SentencesToListOfWords(),
		jiwer.RemoveEmptyStrings(),
	])

	return jiwer.wer(references, predictions, truth_transform=transformation, hypothesis_transform=transformation)


def corpus_cer(references: List[str], predictions: List[str]):
	if len(references) != len(predictions):
		raise ValueError("The lists must have the same length")

	total = 0
	for ref, pred in zip(references, predictions):
		total = total + sentence_cer(pred, ref)
	return total / len(references)
