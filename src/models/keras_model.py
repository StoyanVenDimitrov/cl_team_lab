"""NN model implementation with keras"""
import tensorflow as tf
from src.models.model import Model
from tensorflow.keras import backend as K
import tensorflow_hub as hub
import bert


from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

BUFFER_SIZE = 11000
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
FullTokenizer = bert.bert_tokenization.FullTokenizer
max_seq_length = 511  # Your choice here.


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config):#, vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        # self.create_model(
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.batch_size = int(config['batch_size'])
        self.number_of_epochs = int(config['number_of_epochs'])
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
        # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
        #                             trainable=True)
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
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        label_output = tf.keras.layers.Dense(labels_size+1, activation="softmax")(
            state_h
        )
        section_output = tf.keras.layers.Dense(section_size+1, activation="softmax")(
            state_h
        )
        worthiness_output = tf.keras.layers.Dense(worthiness_size+1, activation="softmax")(
            state_h
        )

        self.model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids],
            outputs=[label_output, section_output, worthiness_output]
        )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )
        # for the loss object: https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/

        def masked_loss_function(y_true, y_pred):
            # target: A tensor with the same shape as output.
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            return K.binary_crossentropy(y_v * mask, y_pred * mask)

        # masked_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=masked_loss_function, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def fit_model(self, dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)

        self.model.fit(dataset, epochs=self.number_of_epochs)

    def prepare_output_data(self, data):
        """tokenize and encode for label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter('__unknown__'.__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        # TODO: pad batches, not the whole set
        return tensor, component_tokenizer

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