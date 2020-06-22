"""NN model implementation with keras"""
import tensorflow as tf
import tensorflow_datasets as tfds
from src.models.model import Model
from tensorflow.keras import backend as K

from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config, vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        self.worthiness_encoder = tfds.features.text.TokenTextEncoder(2)
        self.create_model(
            config["embedding_dim"],
            config["rnn_units"],
            vocab_size,
            labels_size,
            section_size,
            worthiness_size
        )
        self.mask_value = -1  # for masking missing labels

    def create_model(
        self, embedding_dim, rnn_units, vocab_size, labels_size, section_size, worthiness_size
    ):
        text_input_layer = tf.keras.Input(
            shape=(None,), dtype=tf.string, name="Input_1"
        )
        embeddings_layer = tf.keras.layers.Embedding(
            vocab_size + 1, embedding_dim, mask_zero=True
        )

        input_embedding = embeddings_layer(text_input_layer)
        output, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                rnn_units,
                stateful=False,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        )(
            input_embedding
        )
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        label_output = tf.keras.layers.Dense(labels_size + 1, activation="softmax")(
            state_h
        )
        section_output = tf.keras.layers.Dense(section_size + 1, activation="softmax")(
            state_h
        )
        worthiness_output = tf.keras.layers.Dense(worthiness_size + 1, activation="softmax")(
            state_h
        )


        self.model = tf.keras.Model(
            inputs=text_input_layer, outputs=[label_output, section_output, worthiness_output]
        )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )
    # https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
