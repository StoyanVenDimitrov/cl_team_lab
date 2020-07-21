"""NN model implementation with keras"""
import tensorflow as tf
from src.models.model import Model
from tensorflow.keras import backend as K
import tensorflow_hub as hub
import tensorflow_addons as tfa
import bert


from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

BUFFER_SIZE = 11000
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
FullTokenizer = bert.bert_tokenization.FullTokenizer
max_seq_length = 100  # Your choice here.


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config):#, vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        # self.create_model(
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.atention_size = 2 * self.rnn_units
        self.batch_size = int(config['batch_size'])
        self.number_of_epochs = int(config['number_of_epochs'])
        self.validation_step = int(config['validation_step'])
        self.mask_value = 1  # for masking missing label

    def create_model(
        self, labels_size, section_size, worthiness_size
    ):
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        output, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.rnn_units,
                stateful=False,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        )(
            sequence_output
        )
        # state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_h = WeirdAttention(self.atention_size)(output)
        label_output = tf.keras.layers.Dense(labels_size+1, activation="softmax")(
            state_h
        )
        section_output = tf.keras.layers.Dense(section_size+1, activation="softmax")(
            state_h
        )
        worthiness_output = tf.keras.layers.Dense(worthiness_size+1, activation="softmax")(
            state_h
        )

        self.our_model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids],
            outputs=[label_output, section_output, worthiness_output]
        )

        self.our_model.summary()
        tf.keras.utils.plot_model(
            self.our_model, to_file="multi_input_and_output_model.png", show_shapes=True
        )

        def masked_loss_function(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            return K.categorical_crossentropy(
                y_v * mask,
                K.clip(y_pred * mask, min_value=1e-15, max_value=1e10)
            )

        # masked_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.our_model.compile(
            optimizer='adam',
            loss=masked_loss_function,
            metrics={'dense': F1ForMultitask(num_classes=labels_size)} #, average='macro')}
        )

    def fit_model(self, dataset, val_dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        #val_batch_size = tf.data.experimental.cardinality(val_dataset).numpy()
        val_dataset = val_dataset.padded_batch(5, drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)
        self.our_model.fit(
            dataset,
            epochs=self.number_of_epochs,
            callbacks=[ValidateAfter(val_dataset, self.validation_step)]
        )

    def prepare_output_data(self, data):
        """tokenize and encode for label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter('__unknown__'.__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor, component_tokenizer
    
    def prepare_dev_output_data(self, data, tokenizer):
        tensor = tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor

    def prepare_input_data(self, data):
        """prepare text input for BERT"""
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)
        input_ids, input_masks, input_segments = [], [], []

        for s in data:
            stokens = tokenizer.tokenize(s)
            stokens = ["[CLS]"] + stokens + ["[SEP]"]
            input_ids.append(get_ids(stokens, tokenizer, max_seq_length))
            input_masks.append(get_masks(stokens, max_seq_length))
            input_segments.append(get_segments(stokens, max_seq_length))
        return input_ids, input_masks, input_segments

    def create_dataset(self, ids, mask, segments, labels, sections, worthiness):
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    'input_word_ids': ids,
                    'input_mask': mask,
                    'segment_ids': segments
                },
                {
                    'dense': labels,
                    'dense_1': sections,
                    'dense_2': worthiness
                }
            )
        )
        return dataset

    def create_dev_dataset(self, ids, mask, segments, labels):
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    'input_word_ids': ids,
                    'input_mask': mask,
                    'segment_ids': segments
                },
                {
                    'dense': labels
                }
            )
        )
        return dataset


class SingletaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config):#, vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        # self.create_model(
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.atention_size = 2 * self.rnn_units
        self.batch_size = int(config['batch_size'])
        self.number_of_epochs = int(config['number_of_epochs'])
        self.mask_value = 1  # for masking missing label
        self.validation_step = int(config['validation_step'])

    def create_model(
        self, labels_size
    ):
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        output, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.rnn_units,
                stateful=False,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        )(
            sequence_output
        )
        # state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_h = WeirdAttention(self.atention_size)(output)
        label_output = tf.keras.layers.Dense(labels_size+1, activation="softmax")(
            state_h
        )

        self.model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=label_output
        )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="single_input_and_output_model.png", show_shapes=True
        )

        def loss_function(y_true, y_pred):
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=False
            )
            return loss

        self.model.compile(
            optimizer='adam',
            loss=loss_function,
            metrics={'dense': F1ForMultitask(num_classes=labels_size)}  # , average='macro')}
        )

    def fit_model(self, dataset, val_dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.padded_batch(2, drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)
        self.model.fit(
            dataset,
            epochs=self.number_of_epochs,
            callbacks=[ValidateAfter(val_dataset, self.validation_step)]
        )

    def prepare_output_data(self, data):
        """tokenize and encode for label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter('__unknown__'.__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor, component_tokenizer
    
    def prepare_dev_output_data(self, data, tokenizer):
        tensor = tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor

    def prepare_input_data(self, data):
        """prepare text input for BERT"""
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)
        input_ids, input_masks, input_segments = [], [], []

        for s in data:
            stokens = tokenizer.tokenize(s)
            stokens = ["[CLS]"] + stokens + ["[SEP]"]
            input_ids.append(get_ids(stokens, tokenizer, max_seq_length))
            input_masks.append(get_masks(stokens, max_seq_length))
            input_segments.append(get_segments(stokens, max_seq_length))
        return input_ids, input_masks, input_segments

    def create_dataset(self, ids, mask, segments, labels):
        dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        'input_word_ids': ids,
                        'input_mask': mask,
                        'segment_ids': segments
                    },
                    {
                        'dense': labels
                    }
                )
            )
        return dataset

    def create_dev_dataset(self, ids, mask, segments, labels):
            return self.create_dataset(ids, mask, segments, labels)

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        return [1] * max_seq_length
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    if len(tokens) > max_seq_length:
        return [0]*max_seq_length
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(tokens) > max_seq_length:
        return token_ids[:max_seq_length]
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


class WeirdAttention(tf.keras.layers.Layer):
    """attention as in Cohan et al., 2019"""
    def __init__(self, units):
        super(WeirdAttention, self).__init__()
        self.units = units
        self.w = self.add_weight(shape=(self.units, 1),
                                 initializer='random_normal',
                                 trainable=True)

    # TODO: strictly, self.w should be added in the build():
    # https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
    def call(self, inputs):
        alpha_score = tf.linalg.matvec(inputs, tf.squeeze(self.w))
        alpha = K.softmax(alpha_score)
        scored_input = inputs * tf.expand_dims(alpha, axis=-1)
        return tf.reduce_sum(scored_input, 1)


class ValidateAfter(tf.keras.callbacks.Callback):
    def __init__(self, val_data, val_step):
        super(ValidateAfter, self).__init__()
        self.val_data = val_data,
        self.val_step = val_step

    def on_train_batch_end(self, batch, logs=None):
        # if batch > 1 and batch % self.val_step == 0:
        #     self.model.evaluate(self.val_data[0], verbose=1)
        self.model.evaluate(self.val_data[0], verbose=1)


class F1ForMultitask(tfa.metrics.F1Score):
    def __init__(
        self,
        num_classes,
        average=None,
        threshold=None,
        name: str = "f1_score",
        dtype=None,
    ):
        super().__init__(num_classes-1, average, threshold, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # skip to count samples with label __unknown__
        mask = K.cast(K.not_equal(y_true, 1), K.floatx())
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold
        y_true = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        # # skip counting samples where the PAD token is predicted
        # mask_padding = K.expand_dims(K.cast(K.not_equal(y_pred[:,0], 1), K.floatx()), -1)
        # y_pred = y_pred * mask_padding

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val*mask, axis=self.axis)[2:]
            # return tf.reduce_sum(val, axis=self.axis)
        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))