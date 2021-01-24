import tensorflow as tf


class CTCLayer(tf.keras.layers.Layer):
	def __init__(self, blank_index=None, logits_time_major=True, unique=None, **kwargs):
		super(CTCLayer, self).__init__(**kwargs)
		self.loss_fn = tf.nn.ctc_loss
		self.blank_index = blank_index
		self.logits_time_major = logits_time_major
		self.unique = unique

	def call(self, y_true, y_pred, input_length, label_length):
		# Compute the training-time loss value and add it
		# to the layer using `self.add_loss()`.

		y_true = tf.cast(y_true, tf.int32)
		y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]))
		input_length = tf.reshape(input_length, (-1,))
		label_length = tf.reshape(label_length, (-1,))

		loss = self.loss_fn(y_true, y_pred, label_length, input_length,
		                    logits_time_major=self.logits_time_major,
		                    unique=self.unique,
		                    blank_index=self.blank_index)
		self.add_loss(tf.reduce_mean(loss))

		# At test time, just return the computed predictions
		return y_pred
