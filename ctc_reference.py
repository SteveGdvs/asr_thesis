import tensorflow as tf

from thesis_lib.data.common_voice import read_tsv
from thesis_lib.ds_utils.ds_map_functions import to_mfccs, to_ctc_format
from thesis_lib.ds_utils.helpers import split_ds
from thesis_lib.models import create_ctc_model, get_prediction_model
from thesis_lib.preprocessing.common_voice import filter_cv, preprocess_cv_ctc
from thesis_lib.preprocessing.utils import get_string_lookup, decode_ctc_batch_predictions, reverse_tokenization

dataset_path = "common_voice_dataset_path/"
clips_path = dataset_path + "clips/"

final_tsv = read_tsv(dataset_path, "train.tsv") + read_tsv(dataset_path, "dev.tsv") + read_tsv(dataset_path, "test.tsv")
print("final_tsv len: {0}".format(len(final_tsv)))

#
character_level = True
to_lower = False
to_ascii = True
min_duration = 2
max_duration = 6

filter_tsv = filter_cv(tsv=final_tsv,
                       clips_path=clips_path,
                       min_duration=min_duration,
                       max_duration=max_duration)

data, vocab, max_sentence_length = preprocess_cv_ctc(tsv=filter_tsv,
                                                     clips_path=clips_path,
                                                     character_level=character_level,
                                                     to_lower=to_lower,
                                                     to_ascii=to_ascii)

vocab_size = len(vocab)
print("Number of sentences:", len(data))
print("Number of unique output tokens:", vocab_size)
print("Max sentences length for outputs:", max_sentence_length)

vocab_to_num, num_to_vocab = get_string_lookup(vocab)
#
dataset = tf.data.Dataset.from_tensor_slices(data)

val_percentage = 0.2
test_percentage = 0.2

train_ds, val_ds, test_ds = split_ds(dataset, val_percentage=val_percentage, test_percentage=test_percentage)
train_ds = train_ds.shuffle(64 * 64)
val_ds = val_ds.shuffle(64 * 64)
test_ds = test_ds.shuffle(64 * 64)
#
batch_size = 128
n_mfccs = 13

train_ds_oh = train_ds.map(lambda x: to_mfccs(x, n_mfccs, vocab_to_num, character_level))
train_ds_oh = train_ds_oh.cache()
train_ds_oh = train_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=(-1., -1, -1, -1))
train_ds_oh = train_ds_oh.map(to_ctc_format)
train_ds_oh = train_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

val_ds_oh = val_ds.map(lambda x: to_mfccs(x, n_mfccs, vocab_to_num, character_level))
val_ds_oh = val_ds_oh.cache()
val_ds_oh = val_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=(-1., -1, -1, -1))
val_ds_oh = val_ds_oh.map(to_ctc_format)
val_ds_oh = val_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

test_ds_oh = test_ds.map(lambda x: to_mfccs(x, n_mfccs, vocab_to_num, character_level))
test_ds_oh = test_ds_oh.cache()
test_ds_oh = test_ds_oh.padded_batch(batch_size, drop_remainder=True, padding_values=(-1., -1, -1, -1))
test_ds_oh = test_ds_oh.map(to_ctc_format)
test_ds_oh = test_ds_oh.prefetch(tf.data.experimental.AUTOTUNE)

print("train_ds length: {0} batches".format(len(train_ds_oh)))
print("val_ds length: {0} batches".format(len(val_ds_oh)))
print("test_ds length: {0} batches".format(len(test_ds_oh)))

input_dim = test_ds_oh.element_spec['features_input'].shape[2]
print("input_dim: {0}".format(input_dim))
#

rnn_type = "bi_lstm"
rnn_dims = [128]  # [128, 512, 1024]

epochs = 1

callbacks = [
	tf.keras.callbacks.ReduceLROnPlateau(patience=10, min_delta=1e-1, min_lr=1e-4, verbose=1),
	tf.keras.callbacks.EarlyStopping(patience=15, min_delta=1e-2, verbose=1)
]

ctc_model = create_ctc_model(rnn_type, rnn_dims, input_dim=input_dim, output_dim=vocab_size)

ctc_model.fit(train_ds_oh, epochs=epochs, validation_data=val_ds_oh, callbacks=callbacks)

ctc_model.evaluate(test_ds_oh)

pred_ctc_model = get_prediction_model(ctc_model)

for i in train_ds_oh.take(1):
	preds = pred_ctc_model.predict(i["features_input"])
	targets = reverse_tokenization(i["labels_input"], num_to_vocab, character_level)
	res = decode_ctc_batch_predictions(preds, num_to_vocab, i["features_len"], character_level)
	for i, t in zip(res, targets):
		print(i)
		print(t)
		print()