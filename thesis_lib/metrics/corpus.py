from typing import List

from nltk.translate import bleu_score
from nltk.translate import gleu_score



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