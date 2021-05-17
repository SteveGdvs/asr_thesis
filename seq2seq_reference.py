import tensorflow as tf

from thesis_lib.data.common_voice import read_tsv
from thesis_lib.ds_utils.ds_map_functions import to_seq2seq_format, to_spectrogram, to_one_hot_decoder_only
from thesis_lib.ds_utils.helpers import split_ds
from thesis_lib.models import OneHotSeq2Seq
from thesis_lib.preprocessing.common_voice import preprocess_cv_s2s, filter_cv
from thesis_lib.preprocessing.constants import CHARACTER_PAD_TOKEN, WORD_PAD_TOKEN
from thesis_lib.preprocessing.utils import get_string_lookup

dataset_path = "common_voice_dataset_path/"
clips_path = dataset_path + "clips/"

final_tsv = read_tsv(dataset_path, "train.tsv") + read_tsv(dataset_path, "dev.tsv") + read_tsv(dataset_path, "test.tsv")
print("final_tsv len: {0}".format(len(final_tsv)))

character_level = True
to_lower = False
to_ascii = True
min_duration = None
max_duration = None

filter_tsv = filter_cv(tsv=final_tsv,
                       clips_path=clips_path,
                       min_duration=min_duration,
                       max_duration=max_duration)

data, vocab, max_sentence_length = preprocess_cv_s2s(tsv=filter_tsv,
                                                     clips_path=clips_path,
                                                     character_level=character_level,
                                                     to_lower=to_lower,
                                                     to_ascii=to_ascii)

vocab_size = len(vocab)

print("Number of sentences:", len(data))
print("Number of unique output tokens:", vocab_size)
print("Max sentences length for outputs:", max_sentence_length)

vocab_to_num, num_to_vocab = get_string_lookup(vocab)

dataset = tf.data.Dataset.from_tensor_slices(data)

val_percentage = 0.2
test_percentage = 0.2

train_ds, val_ds, test_ds = split_ds(dataset, val_percentage=val_percentage, test_percentage=test_percentage)
train_ds = train_ds.shuffle(64 * 64)
val_ds = val_ds.shuffle(64 * 64)
test_ds = test_ds.shuffle(64 * 64)

i_pad_value = 0.
if character_level:
	t_pad_value = vocab_to_num(CHARACTER_PAD_TOKEN).numpy().astype("int32")
else:
	t_pad_value = vocab_to_num(WORD_PAD_TOKEN).numpy().astype("int32")

batch_size = 128

train_ds_oh = train_ds.map(lambda x: to_spectrogram(x, vocab_to_num, character_level))
train_ds_oh = train_ds_oh.map(lambda x, y, *_: to_seq2seq_format(x, y, t_pad_value))
train_ds_oh = train_ds_oh.cache()
train_ds_oh = train_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
train_ds_oh = train_ds_oh.map(lambda x, y: to_one_hot_decoder_only(x, y, vocab_size))
train_ds_oh = train_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

val_ds_oh = val_ds.map(lambda x: to_spectrogram(x, vocab_to_num, character_level))
val_ds_oh = val_ds_oh.map(lambda x, y, *_: to_seq2seq_format(x, y, t_pad_value))
val_ds_oh = val_ds_oh.cache()
val_ds_oh = val_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
val_ds_oh = val_ds_oh.map(lambda x, y: to_one_hot_decoder_only(x, y, vocab_size))
val_ds_oh = val_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

test_ds_oh = test_ds.map(lambda x: to_spectrogram(x, vocab_to_num, character_level))
test_ds_oh = test_ds_oh.map(lambda x, y, *_: to_seq2seq_format(x, y, t_pad_value))
test_ds_oh = test_ds_oh.cache()
test_ds_oh = test_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
test_ds_oh = test_ds_oh.map(lambda x, y: to_one_hot_decoder_only(x, y, vocab_size))
test_ds_oh = test_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

print("train_ds length: {0} batches".format(len(train_ds_oh)))
print("val_ds length: {0} batches".format(len(val_ds_oh)))
print("test_ds length: {0} batches".format(len(test_ds_oh)))

input_dim = test_ds_oh.element_spec[0][0].shape[2]
print("input_dim: {0}".format(input_dim))

rnn_type = "bi_lstm"
rnn_size = 128  # [128, 512, 1024]

epochs = 250

callbacks = [
	tf.keras.callbacks.ReduceLROnPlateau(patience=10, min_delta=1e-3, min_lr=1e-4, verbose=1),
	tf.keras.callbacks.EarlyStopping(patience=15, min_delta=1e-4, verbose=1)
]

one_hot_model = OneHotSeq2Seq(
	rnn_type=rnn_type,
	rnn_size=rnn_size,
	num_encoder_tokens=input_dim,
	num_decoder_tokens=vocab_size,
	target_token_to_index=vocab_to_num,
	target_index_to_token=num_to_vocab,
	max_decoder_seq_length=max_sentence_length,
	character_level=character_level
)
one_hot_model.create_model()

one_hot_model.fit(train_ds_oh, epochs, val_ds_oh, callbacks)

one_hot_model.plot_history()
one_hot_model.evaluate(test_ds_oh)
