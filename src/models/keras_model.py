"""NN model implementation with keras"""
import warnings
import os

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # ignores warnings about future version of numpy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from src.models.model import Model
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from official.nlp.bert import tokenization
import numpy as np
import json
import bert
from official.nlp.bert import tokenization

from src import evaluation
from src.utils import utils

BUFFER_SIZE = 11000


class MultitaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(
        self, config
    ):  # , vocab_size, labels_size, section_size, worthiness_size):
        pre_config = config["preprocessor"]
        config = config["multitask_trainer"]
        super().__init__(config)
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.attention_size = 2 * self.rnn_units
        self.batch_size = int(config["batch_size"])
        self.number_of_epochs = int(config["number_of_epochs"])
        self.mask_value = 1  # for masking missing label
        self.max_seq_len = int(config["max_len"])
        self.use_attention = True if config["use_attention"] == "True" else False
        self.validation_step = int(config["validation_step"])
        self.worthiness_weight = float(config["worthiness_weight"])
        self.section_weight = float(config["section_weight"])

        self.embedding_type = config["embedding_type"]

        if self.embedding_type == "elmo":
            print("Loading ELMo embedding...")
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="tokens", output_key="elmo", trainable=False)
        elif self.embedding_type == "bert":
            print("Loading BERT embedding...")
            # self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            #                 trainable=False)
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1",
                            trainable=False)
        elif self.embedding_type == "albert":
            print("Loading AlBERT embedding...")
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/1",
                            trainable=False)

        self.logdir = utils.make_logdir("keras", "Multitask", pre_config, config)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)

        checkpoint_path = os.path.join(
            self.logdir, os.pardir, "checkpoints/cp-{epoch:04d}.ckpt"
        )
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=500
        )


    def elmo_embedding(self, x):
        return self.elmo_layer(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tf.constant(int(self.batch_size)*[self.max_seq_len])})

    def create_model(self, vocab_size, labels_size, section_size, worthiness_size):
        self.labels_size, self.section_size, self.worthiness_size = (
            labels_size + 1,
            section_size + 1,
            worthiness_size + 1,
        )
        if self.embedding_type == "lstm":
            text_input_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name="Input_1")
            embeddings_layer = tf.keras.layers.Embedding(
                vocab_size, self.embedding_dim, mask_zero=True
            )

            input_embedding = embeddings_layer(text_input_layer)
            (
                output,
                forward_h,
                forward_c,
                backward_h,
                backward_c,
            ) = tf.keras.layers.Bidirectional(
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
            if self.use_attention:
                state_h = WeirdAttention(self.attention_size)(output)
            else:
                state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
                
        elif self.embedding_type in ["bert", "albert"]:

            input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                               name="input_word_ids")
            input_mask = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                            name="input_mask")
            segment_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                                name="segment_ids")
            pooled_output, sequence_output = self.embedding_layer([input_word_ids, input_mask, segment_ids])

            state_h = pooled_output

        elif self.embedding_type == "elmo":
            text_input_layer = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype="string")
            # input_embedding = tf.keras.layers.Lambda(self.elmo_embedding, output_shape=(self.max_seq_len, 1024))(text_input_layer)
            input_embedding = self.elmo_embedding(text_input_layer)

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
            if self.use_attention:
                state_h = WeirdAttention(self.attention_size)(output)
            else:
                state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])


        label_output = tf.keras.layers.Dense(
            labels_size + 1, activation="softmax", name="dense"
        )(state_h)
        section_output = tf.keras.layers.Dense(
            section_size + 1, activation="softmax", name="dense_1"
        )(state_h)
        worthiness_output = tf.keras.layers.Dense(
            worthiness_size + 1, activation="softmax", name="dense_2"
        )(state_h)
        
        if self.embedding_type == "lstm":
            self.model = tf.keras.Model(
                inputs=text_input_layer,
                outputs=[label_output, section_output, worthiness_output],
            )
        elif self.embedding_type in ["bert", "albert"]:
            self.model = tf.keras.Model(
                inputs=[input_word_ids, input_mask, segment_ids],
                outputs=[label_output, section_output, worthiness_output]
            )

        elif self.embedding_type == "elmo":
            self.model = tf.keras.Model(
                inputs=text_input_layer,
                outputs=[label_output, section_output, worthiness_output],
            )

        self.model.summary()
        tf.keras.utils.plot_model(
            self.model, to_file="multi_input_and_output_model.png", show_shapes=True
        )

        def _masked_loss_function(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            return K.categorical_crossentropy(
                y_v * mask, K.clip(y_pred * mask, min_value=1e-15, max_value=1e10)
            )

        def masked_loss_function(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
            y_v = K.one_hot(K.cast(K.flatten(y_true), tf.int32), y_pred.shape[1])
            # compare y_v len to find out the current type of labels and output the loss coeff or 1:
            section_loss_weight = K.switch(
                K.equal(y_v.shape[1], self.section_size), self.section_weight, 1.0
            )
            worthiness_loss_weight = K.switch(
                K.equal(y_v.shape[1], self.worthiness_size), self.worthiness_weight, 1.0
            )
            # update the mask to scale with the needed factor:
            mask = mask * section_loss_weight * worthiness_loss_weight
            return K.categorical_crossentropy(
                y_v * mask, K.clip(y_pred * mask, min_value=1e-15, max_value=1e10)
            )

        self.model.compile(
            optimizer="adam",
            loss=masked_loss_function,
            metrics={
                "dense": F1ForMultitask(num_classes=labels_size)
            },  # , average='macro')}
        )

    def fit_model(self, dataset, val_dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.padded_batch(2, drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)
        self.model.run_eagerly = True  # solves problem of converting tensor to numpy array https://github.com/tensorflow/tensorflow/issues/27519
        self.model.fit(
            dataset,
            epochs=self.number_of_epochs,
            # callbacks=[
            #     ValidateAfter(val_dataset, self.validation_step),
                # self.tensorboard_callback,
                # self.checkpoint_callback,
            # ],
        )

    def eval(self, dataset, save_output=True):
        batch_dataset = dataset.padded_batch(self.batch_size, drop_remainder=False)

        preds = self.model.predict(batch_dataset)
        y_pred = preds[0][:, 2:].argmax(1) + 2
        y_true = [l["dense"][0].numpy() for _, l in dataset.take(-1)]

        report_json = classification_report(
            np.asarray(y_true), y_pred, labels=[2, 3, 4], output_dict=True
        )
        report_text = classification_report(
            np.asarray(y_true), y_pred, labels=[2, 3, 4]
        )
        print(report_text)

        if save_output:
            results_path = os.path.join(self.logdir, os.pardir)
            with open(results_path + "/results.json", "w") as f:
                json.dump(report_json, f)
            with open(results_path + "/results.txt", "w") as f:
                f.write(report_text)
            print("Saved result files results.json and results.txt to:", results_path)

    def prepare_data(self, data, max_len=None):
        """tokenize and encode for text, label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter("__unknown__".__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
        if max_len:
            tensor = tensor[:, :max_len]
        # TODO: pad batches, not the whole set
        return tensor, component_tokenizer

    def prepare_dev_data(self, data, tokenizer, max_len=None):
        tensor = tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
        if max_len:
            tensor = tensor[:, :max_len]
        return tensor

    def prepare_input_data(self, data):
        """prepare text input for BERT and Albert"""
        if self.embedding_type == "bert":
            vocab_file = self.embedding_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = self.embedding_layer.resolved_object.do_lower_case.numpy()
            tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
        elif self.embedding_type == "albert":
            sp_model_file = self.embedding_layer.resolved_object.sp_model_file.asset_path.numpy()
            tokenizer = tokenization.FullSentencePieceTokenizer(sp_model_file)

        input_ids, input_masks, input_segments = [], [], []

        for s in data:
            stokens = tokenizer.tokenize(s)
            stokens = ["[CLS]"] + stokens + ["[SEP]"]
            input_ids.append(self.get_ids(stokens, tokenizer))
            input_masks.append(self.get_masks(stokens))
            input_segments.append(self.get_segments(stokens))
        return input_ids, input_masks, input_segments

    def get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens)>self.max_seq_len:
            return [1] * self.max_seq_len
        return [1]*len(tokens) + [0] * (self.max_seq_len - len(tokens))

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        if len(tokens) > self.max_seq_len:
            return [0]*self.max_seq_len
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.max_seq_len - len(tokens))

    def get_ids(self, tokens, tokenizer):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > self.max_seq_len:
            return token_ids[:self.max_seq_len]
        input_ids = token_ids + [0] * (self.max_seq_len-len(token_ids))
        return input_ids

    def create_dataset(self, text, labels, sections, worthiness, ids=None, mask=None, segments=None):
        if self.embedding_type in "lstm":
            dataset = tf.data.Dataset.from_tensor_slices(
                (text, {"dense": labels, "dense_1": sections, "dense_2": worthiness})
            )
        elif self.embedding_type in ["bert", "albert"]:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        'input_word_ids': ids,
                        'input_mask': mask,
                        'segment_ids': segments
                    },
                    {
                        'dense': labels,
                        # 'dense_1': sections,
                        # 'dense_2': worthiness
                    }
                )
            )
        elif self.embedding_type == "elmo":
            dataset = tf.data.Dataset.from_tensor_slices(
                (text, {"dense": labels, "dense_1": sections, "dense_2": worthiness})
            )
        return dataset

    def create_dev_dataset(self, text, labels, ids=None, mask=None, segments=None):
        if self.embedding_type == "lstm":
            dataset = tf.data.Dataset.from_tensor_slices((text, {"dense": labels}))
        elif self.embedding_type in ["bert", "albert"]:
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
        elif self.embedding_type == "elmo":
            dataset = tf.data.Dataset.from_tensor_slices((text, {"dense": labels}))
        return dataset

    # TODO model.save throws error https://github.com/tensorflow/tensorflow/issues/26811
    def save_model(self):
        path = os.path.join(self.logdir, os.pardir, "model.h5")
        print("Saving model to path:", path)
        # self.model.save(path)
        self.model.save_weights(path)


class SingletaskLearner(Model):
    """Multitask learning environment for citation classification (main task) and citation section title (auxiliary)"""

    def __init__(
        self, config
    ):  # , vocab_size, labels_size, section_size, worthiness_size):
        pre_config = config["preprocessor"]
        config = config["singletask_trainer"]
        super().__init__(config)
        self.embedding_dim = int(config["embedding_dim"])
        self.rnn_units = int(config["rnn_units"])
        self.attention_size = 2 * self.rnn_units
        self.batch_size = int(config["batch_size"])
        self.number_of_epochs = int(config["number_of_epochs"])
        self.mask_value = 1  # for masking missing label
        self.max_seq_len = int(config["max_len"])
        self.use_attention = True if config["use_attention"] == "True" else False
        self.validation_step = int(config["validation_step"])

        self.embedding_type = config["embedding_type"]

        if self.embedding_type == "elmo":
            print("Loading ELMo embedding...")
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="tokens", output_key="elmo", trainable=False)
        elif self.embedding_type == "bert":
            # self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            #                 trainable=False)
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1",
                            trainable=False)
        elif self.embedding_type == "albert":
            self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_large/1",
                            trainable=False)
        elif self.embedding_type == "albert":
            self.albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_large/1",
                            trainable=False)

        self.logdir = utils.make_logdir("keras", "Singletask", pre_config, config)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)

        checkpoint_path = os.path.join(
            self.logdir, os.pardir, "checkpoints/cp-{epoch:04d}.ckpt"
        )
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=500
        )

    def elmo_embedding(self, x):
        return self.elmo_layer(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tf.constant(int(self.batch_size)*[self.max_seq_len])})

    def create_model(self, vocab_size, labels_size):
        
        if self.embedding_type == "lstm":
            text_input_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name="Input_1")
            embeddings_layer = tf.keras.layers.Embedding(
                vocab_size, self.embedding_dim, mask_zero=True
            )

            input_embedding = embeddings_layer(text_input_layer)
            (
                output,
                forward_h,
                forward_c,
                backward_h,
                backward_c,
            ) = tf.keras.layers.Bidirectional(
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
            if self.use_attention:
                state_h = WeirdAttention(self.attention_size)(output)
            else:
                state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
                
        elif self.embedding_type in ["bert", "albert"]:
            # if self.embedding_type == "bert":
            #     self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            #                     trainable=True)
            # else:
            #     self.embedding_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_large/1",
            #                   trainable=True)

            input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                               name="input_word_ids")
            input_mask = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                            name="input_mask")
            segment_ids = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32,
                                                name="segment_ids")
            pooled_output, sequence_output = self.embedding_layer([input_word_ids, input_mask, segment_ids])

            state_h = pooled_output            
        
        elif self.embedding_type == "elmo":
            text_input_layer = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype="string")
            input_embedding = tf.keras.layers.Lambda(self.elmo_embedding, output_shape=(self.max_seq_len, 1024))(text_input_layer)

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
            if self.use_attention:
                state_h = WeirdAttention(self.attention_size)(output)
            else:
                state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        
        label_output = tf.keras.layers.Dense(labels_size + 1, activation="softmax")(
            state_h
        )



        if self.embedding_type == "lstm":
            self.model = tf.keras.Model(
                inputs=text_input_layer,
                outputs=label_output
            )
        elif self.embedding_type in ["bert", "albert"]:
            self.model = tf.keras.Model(
                inputs=[input_word_ids, input_mask, segment_ids],
                outputs=label_output
            )

        elif self.embedding_type == "albert":
            self.model = tf.keras.Model(
                inputs=[input_word_ids, input_mask, segment_ids],
                outputs=label_output
            )

        elif self.embedding_type == "elmo":
            self.model = tf.keras.Model(
                inputs=text_input_layer,
                outputs=label_output
            )

        # self.model = tf.keras.Model(inputs=text_input_layer, outputs=label_output)

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
            optimizer="adam",
            loss=loss_function,
            metrics={
                "dense": F1ForMultitask(num_classes=labels_size)
            },  # , average='macro')}
        )

    def fit_model(self, dataset, val_dataset):
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.padded_batch(2, drop_remainder=True)
        dataset = dataset.shuffle(BUFFER_SIZE)
        self.model.run_eagerly = True  # solves problem of converting tensor to numpy array https://github.com/tensorflow/tensorflow/issues/27519
        self.model.fit(
            dataset,
            epochs=self.number_of_epochs,
            callbacks=[
                ValidateAfter(val_dataset, self.validation_step),
                self.tensorboard_callback,
                self.checkpoint_callback,
            ],
        )

    def eval(self, dataset, save_output=True):
        batch_dataset = dataset.padded_batch(self.batch_size, drop_remainder=False)

        preds = self.model.predict(batch_dataset)
        y_pred = preds[:, 2:].argmax(1) + 2
        y_true = [l["dense"][0].numpy() for _, l in dataset.take(-1)]

        report_json = classification_report(
            np.asarray(y_true), y_pred, labels=[2, 3, 4], output_dict=True
        )
        report_text = classification_report(
            np.asarray(y_true), y_pred, labels=[2, 3, 4]
        )
        print(report_text)

        if save_output:
            results_path = os.path.join(self.logdir, os.pardir)
            with open(results_path + "/results.json", "w") as f:
                json.dump(report_json, f)
            with open(results_path + "/results.txt", "w") as f:
                f.write(report_text)
            print("Saved result files results.json and results.txt to:", results_path)

    def prepare_data(self, data, max_len=None):
        """tokenize and encode for text, label, section... """
        component_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=1)

        filtered_data = list(filter("__unknown__".__ne__, data))
        component_tokenizer.fit_on_texts(filtered_data)

        tensor = component_tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
        if max_len:
            tensor = tensor[:, :max_len]
        # TODO: pad batches, not the whole set
        return tensor, component_tokenizer

    def prepare_dev_data(self, data, tokenizer, max_len=None):
        tensor = tokenizer.texts_to_sequences(data)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
        if max_len:
            tensor = tensor[:, :max_len]
        return tensor

    def prepare_input_data(self, data):
        """prepare text input for BERT and Albert"""
        if self.embedding_type == "bert":
            vocab_file = self.embedding_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = self.embedding_layer.resolved_object.do_lower_case.numpy()
            tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
        elif self.embedding_type == "albert":
            sp_model_file = self.embedding_layer.resolved_object.sp_model_file.asset_path.numpy()
            tokenizer = tokenization.FullSentencePieceTokenizer(sp_model_file)

        input_ids, input_masks, input_segments = [], [], []

        for s in data:
            stokens = tokenizer.tokenize(s)
            stokens = ["[CLS]"] + stokens + ["[SEP]"]
            input_ids.append(self.get_ids(stokens, tokenizer))
            input_masks.append(self.get_masks(stokens))
            input_segments.append(self.get_segments(stokens))
        return input_ids, input_masks, input_segments

    def get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens)>self.max_seq_len:
            return [1] * self.max_seq_len
        return [1]*len(tokens) + [0] * (self.max_seq_len - len(tokens))

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        if len(tokens) > self.max_seq_len:
            return [0]*self.max_seq_len
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.max_seq_len - len(tokens))

    def get_ids(self, tokens, tokenizer):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > self.max_seq_len:
            return token_ids[:self.max_seq_len]
        input_ids = token_ids + [0] * (self.max_seq_len-len(token_ids))
        return input_ids


    def create_dataset(self, text, labels, ids=None, mask=None, segments=None):
        if self.embedding_type in ["lstm", "elmo"]:
            dataset = tf.data.Dataset.from_tensor_slices((text, {"dense": labels}))
        elif self.embedding_type in ["bert", "albert"]:
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

    def create_dev_dataset(self, text, labels, ids=None, mask=None, segments=None):
        if self.embedding_type in ["lstm", "elmo"]:
            dataset = tf.data.Dataset.from_tensor_slices((text, {"dense": labels}))
        elif self.embedding_type in ["bert", "albert"]:
            dataset = self.create_dataset(text, labels, ids=ids, mask=mask, segments=segments)
        return dataset

    # TODO model.save throws error https://github.com/tensorflow/tensorflow/issues/26811
    def save_model(self):
        path = os.path.join(self.logdir, os.pardir, "model.h5")
        print("Saving model to path:", path)
        # self.model.save(path)
        self.model.save_weights(path)


class WeirdAttention(tf.keras.layers.Layer):
    """attention as in Cohan et al., 2019"""

    def __init__(self, units):
        super(WeirdAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.units, 1), initializer="random_normal", trainable=True, name="w"
        )

    def call(self, inputs):
        alpha_score = tf.linalg.matvec(inputs, tf.squeeze(self.w))
        alpha = K.softmax(alpha_score)
        scored_input = inputs * tf.expand_dims(alpha, axis=-1)
        return tf.reduce_sum(scored_input, 1)


class ValidateAfter(tf.keras.callbacks.Callback):
    def __init__(self, val_data, val_step):
        super(ValidateAfter, self).__init__()
        self.val_data = (val_data,)
        self.val_step = val_step

    def on_train_batch_end(self, batch, logs=None):
        if batch > 1 and batch % self.val_step == 0:
            self.model.evaluate(self.val_data[0], verbose=1)


class F1ForMultitask(tfa.metrics.F1Score):
    def __init__(
        self,
        num_classes,
        # log_metrics,
        average=None,
        threshold=None,
        name: str = "f1_score",
        dtype=None,
    ):
        super().__init__(num_classes - 1, average, threshold, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # false_idxs = np.concatenate(
        #     (
        #         np.where(y_true.numpy().flatten() == 0)[0],
        #         np.where(y_true.numpy().flatten() == 1)[0],
        #     )
        # )
        # _true = np.delete(y_true.numpy().flatten(), false_idxs)
        # _pred = np.delete(y_pred.numpy()[:, 2:].argmax(1) + 2, false_idxs)
        # report = classification_report(_true, _pred, labels=[2, 3, 4], output_dict=True)
        # if log_metrics:
        #     wandb_log_report(report)
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
            return tf.reduce_sum(val * mask, axis=self.axis)[2:]
            # return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))

        # if self.log_metrics:
        #     wandb_log_report("test", test_report)
