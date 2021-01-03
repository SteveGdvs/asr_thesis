import jiwer
from nltk.metrics import distance
from nltk.translate import bleu_score
from nltk.translate import gleu_score


def sentence_gleu(reference: str, prediction: str):
	return gleu_score.sentence_gleu([reference], prediction)


def sentence_bleu(reference: str, prediction: str, smoothing_function=bleu_score.SmoothingFunction().method1):
	return bleu_score.sentence_bleu([reference], prediction, smoothing_function=smoothing_function)


def sentence_wer(reference: str, prediction: str):
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

	return jiwer.wer(reference, prediction, truth_transform=transformation, hypothesis_transform=transformation)


def sentence_cer(reference: str, prediction: str):
	i_s_d = distance.edit_distance(prediction, reference)
	return i_s_d/len(reference)
