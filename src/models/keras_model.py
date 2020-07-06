"""NN model implementation with keras"""
# import tensorflow as tf
# import tf.keras as keras
# from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
from src.models.model import Model
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras import backend as K
import tensorflow_hub as hub
from nltk.tokenize import WordPunctTokenizer
from src import evaluation

tf.disable_v2_behavior()


"""Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

BUFFER_SIZE = 11000


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(self, config):  # vocab_size, labels_size, section_size, worthiness_size):
        super().__init__(config)
        self.config = config
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.batch_size = int(config['batch_size'])
        self.number_of_epochs = int(config['number_of_epochs'])
        self.mask_value = 1  # for masking missing label
        self.max_len = int(self.config["max_len"])

    def elmo_embedding_v2(self, x):
        elmo_layer = hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="tokens", output_key="elmo", trainable=False)
        embedding = elmo_layer(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tf.constant(int(self.config["batch_size"])*[self.max_len])})
        return embedding

    def create_model(
        self, vocab_size, labels_size, section_size, worthiness_size
    ):
        text_input_layer = keras.layers.Input(shape=(self.max_len,), dtype="string")
        input_embedding = keras.layers.Lambda(self.elmo_embedding_v2, output_shape=(self.max_len, 1024))(text_input_layer)

        output, forward_h, forward_c, backward_h, backward_c = keras.layers.Bidirectional(
            keras.layers.LSTM(
                self.rnn_units,
                stateful=False,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        )(
            input_embedding
        )
        state_h = keras.layers.Concatenate()([forward_h, backward_h])
        label_output = keras.layers.Dense(labels_size+1, activation="softmax")(
            state_h
        )
        section_output = keras.layers.Dense(section_size+1, activation="softmax")(
            state_h
        )
        worthiness_output = keras.layers.Dense(worthiness_size+1, activation="softmax")(
            state_h
        )

        self.model = keras.Model(
            inputs=text_input_layer, outputs=[label_output, section_output, worthiness_output]
        )

        self.model.summary()
        keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )
        # for the loss object: https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/

        def masked_loss_function(y_true, y_pred):
            # target: A tensor with the same shape as output.
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            return K.binary_crossentropy(y_v * mask, y_pred * mask)

        # masked_loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=masked_loss_function, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    def fit_model(self, dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.batch_size, padded_shapes=([None, None]), drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)

        self.model.fit(dataset, epochs=self.number_of_epochs)

    def prepare_data(self, data, typ=None, pad_token=None):
        """tokenize and encode for text, label, section... """
        if not typ:
            component_tokenizer = keras.preprocessing.text.Tokenizer(oov_token=1)
            filtered_data = list(filter('__unknown__'.__ne__, data))
            component_tokenizer.fit_on_texts(filtered_data)
            tensor = component_tokenizer.texts_to_sequences(data)
            tensor = keras.preprocessing.sequence.pad_sequences(tensor,
                                                                   padding='post')
            return tensor, component_tokenizer

        elif typ == "text":
            tokenizer = WordPunctTokenizer()
            tokens = [[str(t) for t in tokenizer.tokenize(sent)] for sent in data]
            tensor = keras.preprocessing.sequence.pad_sequences(tokens, padding="post",
                                                                   dtype=object,
                                                                   maxlen=self.max_len,
                                                                   value=pad_token)
            # TODO: pad batches, not the whole set
            return tensor, list(set([t for s in tokens for t in s]))

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
