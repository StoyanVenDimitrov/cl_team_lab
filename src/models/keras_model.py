"""NN model implementation with keras"""
import tensorflow as tf
import tensorflow_datasets as tfds
from src.models.model import Model
from tensorflow.keras import backend as K

from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

BUFFER_SIZE = 11000
BATCH_SIZE = 24
EPOCHS = 1


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config):#, vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        # self.create_model(
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.mask_value = 1  # for masking missing label

    def create_model(
        self, vocab_size, labels_size, section_size, worthiness_size
    ):
        text_input_layer = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name="Input_1"
        )
        embeddings_layer = tf.keras.layers.Embedding(
            vocab_size, self.embedding_dim, mask_zero=True
        )

        input_embedding = embeddings_layer(text_input_layer)
        output, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.rnn_units,
                stateful=False,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        )(
            input_embedding
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
            inputs=text_input_layer, outputs=[label_output, section_output, worthiness_output]
        )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )
        # for the loss object: https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/

        # def masked_loss_function(y_true, y_pred):
        #     mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
        #     return K.binary_crossentropy(y_true * mask, y_pred * mask)

        masked_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=masked_loss_function, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def fit_model(self, dataset):
        # ds_series_batch = dataset.shuffle(
        #     BUFFER_SIZE,
        #     reshuffle_each_iteration=True).padded_batch(BATCH_SIZE,
        #                                                 padded_shapes=(
        #                                                     [None],
        #                                                     {
        #                                                         'dense': [None],
        #                                                         'dense_1': [None],
        #                                                         'dense_2': [None]
        #                                                     }
        #                                                 ),
        #                                                 drop_remainder=True)
        # self.model.fit(ds_series_batch, epochs=EPOCHS)
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)

        self.model.fit(dataset, epochs=EPOCHS)


    def prepare_data(self, data):
        """tokenize and encode for text, label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter('__unknown__'.__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        # TODO: pad batches, not the whole set
        return tensor, component_tokenizer

    def create_dataset(self, text, labels, sections, worthiness):
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                text,
                {
                    'dense': labels,
                    'dense_1': sections,
                    'dense_2': worthiness
                }
            )
        )
        #dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset
