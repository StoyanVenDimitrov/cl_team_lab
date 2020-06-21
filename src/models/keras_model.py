"""NN model implementation with keras"""
import tensorflow as tf
import tensorflow_datasets as tfds
from src.models.model import Model

from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config, vocab_set, label_set, section_set):
        super().__init__(config)
        self.text_encoder = tfds.features.text.TokenTextEncoder(vocab_set)
        self.label_encoder = tfds.features.text.TokenTextEncoder(label_set)
        self.section_encoder = tfds.features.text.TokenTextEncoder(section_set)
        self.worthiness_encoder = tfds.features.text.TokenTextEncoder(2)
        self.create_model(
            config["vocab_size"],
            config["embedding_dim"],
            config["rnn_units"],
            config["label_size"],
            config["intent_size"],
        )
        self.mask_value = -1  # for masking missing labels

    def create_model(
        self, vocab_size, embedding_dim, rnn_units, label_size, intent_size
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
        slot_output = tf.keras.layers.Dense(label_size + 1)(output)
        intent_output = tf.keras.layers.Dense(intent_size + 1, activation="softmax")(
            state_h
        )

        self.model = tf.keras.Model(
            inputs=text_input_layer, outputs=[slot_output, intent_output]
        )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )


    # https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
