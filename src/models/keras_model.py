"""NN model implementation with keras"""
import tensorflow as tf
from src.models.model import Model
from tensorflow.keras import backend as K

from src import evaluation

"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

BUFFER_SIZE = 11000


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

        def masked_loss_function(y_true, y_pred):
            # target: A tensor with the same shape as output.
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            return K.categorical_crossentropy(y_v * mask, y_pred * mask)

        # masked_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=masked_loss_function, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def fit_model(self, dataset, val_dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.padded_batch(self.batch_size, drop_remainder=True)

        dataset = dataset.shuffle(BUFFER_SIZE)

        self.model.fit(dataset, epochs=self.number_of_epochs, validation_data = val_dataset)


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

    def prepare_dev_data(self, data, tokenizer):
        tensor = tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor

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
        return dataset
