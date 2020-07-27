from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
from src.models.keras_model_tf2_v2 import MultitaskLearner
import tensorflow.compat.v1 as tf
import configparser
import argparse
import time
import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# pylint:skip-file

# config = configparser.ConfigParser()
# config.read("configs/default.conf")


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
        # self.params = params
        # # find the components of the trainer:
        # vectorizer_section = config[config["trainer"]["vectorizer"]]
        # classifier_section = config[config["trainer"]["classifier"]]
        # # find what to instantiate:
        # vectorizer_class = utils.import_module(vectorizer_section["module"])
        # classifier_class = utils.import_module(classifier_section["module"])
        # # instantiate objects of the wished components:
        # self.vectorizer = vectorizer_class(vectorizer_section)
        # self.classifier = classifier_class(classifier_section, self.params)

        # self.reader = SciciteReader(config["trainer"]["dataset"])
        # self.train_set, self.dev_set, self.test_set = self.reader.load_tdt()

    def keras_multitask(self, args):
        start_time = time.time()

        # if self.args.log_metrics:
        #     utils.wandb_init_logs(self.config["multitask_trainer"])

        embedding_type = self.config["multitask_trainer"]["embedding_type"]
        max_len = int(self.config["multitask_trainer"]["max_len"])

        reader = SciciteReader(self.config["preprocessor"])
        print("Loading data...")
        text, labels, sections, worthiness = reader.load_data(_type="train", multitask=True)
        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        text_test, labels_test, _, _ = reader.load_data(_type="test", multitask=False)

        keras_model = MultitaskLearner(self.config)

        if embedding_type == "bert":
           input_ids, input_masks, input_segments = keras_model.prepare_input_data(text)
           dev_input_ids, dev_input_masks, dev_input_segments = keras_model.prepare_input_data(text_dev)
           test_input_ids, test_input_masks, test_input_segments = keras_model.prepare_input_data(text_test)

        print("Preparing data...")
        text_tensor, text_tokenizer = keras_model.prepare_data(text, max_len=max_len)
        labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)
        sections_tensor, sections_tokenizer = keras_model.prepare_data(sections)
        worthiness_tensor, worthiness_tokenizer = keras_model.prepare_data(worthiness)

        text_tensor_dev = keras_model.prepare_dev_data(text_dev, text_tokenizer, max_len=max_len)
        labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)
        text_tensor_test = keras_model.prepare_dev_data(text_test, text_tokenizer, max_len=max_len)
        labels_tensor_test = keras_model.prepare_dev_data(labels_test, labels_tokenizer)

        print("Creating datasets...")
        if embedding_type == "lstm":
            dataset = keras_model.create_dataset(
                text=text_tensor, 
                labels=labels_tensor, 
                sections=sections_tensor, 
                worthiness=worthiness_tensor,
                )
            dev_dataset = keras_model.create_dev_dataset(
                text=text_tensor_dev, 
                labels=labels_tensor_dev)
            test_dataset = keras_model.create_dev_dataset(
                text=text_tensor_test, 
                labels=labels_tensor_test,
                )
        elif embedding_type == "bert":
            dataset = keras_model.create_dataset(
                text=text_tensor,
                labels=labels_tensor,
                sections=sections_tensor,
                worthiness=worthiness_tensor,
                ids=input_ids,
                mask=input_masks,
                segments=input_segments,
            )
            dev_dataset = keras_model.create_dev_dataset(
                text=text_tensor_dev,
                labels=labels_tensor_dev,
                ids=dev_input_ids, 
                mask=dev_input_masks, 
                segments=dev_input_segments,
            )
            test_dataset = keras_model.create_dev_dataset(
                text=text_tensor_test,
                labels=labels_tensor_test,
                ids=test_input_ids, 
                mask=test_input_masks, 
                segments=test_input_segments,
            )
        elif embedding_type == "elmo":
            dataset = keras_model.create_dataset(
                text=text_tensor, 
                labels=labels_tensor, 
                sections=sections_tensor, 
                worthiness=worthiness_tensor,
            )
            dev_dataset = keras_model.create_dev_dataset(
                text=text_tensor_dev, 
                labels=labels_tensor_dev,
                )
            test_dataset = keras_model.create_dev_dataset(
                text=text_tensor_test, 
                labels=labels_tensor_test,
            )

        vocab_size = len(text_tokenizer.word_index.keys())+1
        labels_size = len(labels_tokenizer.word_index.keys())
        section_size = len(sections_tokenizer.word_index.keys())
        worthiness_size = len(worthiness_tokenizer.word_index.keys())

        print("Creating model...")
        keras_model.create_model(vocab_size, labels_size, section_size, worthiness_size)
        print("Fitting model...")
        keras_model.fit_model(dataset, dev_dataset)

        print("Saving model...")
        keras_model.save_model()

        print("Evaluating...")
        keras_model.eval(test_dataset, save_output=True)

        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))

    # def train_feature_extractor(self, training_data=None):
    #     if not training_data:
    #         training_data = self.train_set + self.dev_set + self.test_set
    #     self.vectorizer.generate(training_data)

    # def train_classifier(self, training_data=None, dev_data=None):
    #     if not training_data:
    #         training_data = self.train_set

    #     if not dev_data:
    #         dev_data = self.dev_set

    #     train_set_inputs = [
    #         (
    #             self.vectorizer.vectorize(sample["string"]),
    #             self.vectorizer.vectorize_labels(sample["label"]),
    #         )
    #         for sample in training_data
    #     ]
    #     dev_set_inputs = [
    #         (
    #             self.vectorizer.vectorize(sample["string"]),
    #             self.vectorizer.vectorize_labels(sample["label"]),
    #         )
    #         for sample in dev_data
    #     ]
    #     self.classifier.train(train_set_inputs, dev_set_inputs)
    #     self.statistics = self.classifier.get_train_statistics()

    #     if self.params.log_metrics:
    #         for i in range(len(self.statistics["macro_f1"])):
    #             mlflow.log_metric("Dev Macro F1", self.statistics["macro_f1"][i])
    #             mlflow.log_metric("Dev Micro F1", self.statistics["micro_f1"][i])

    # def evaluate(self):
    #     predicted, labeled = [], []
    #     for sample in self.test_set:
    #         input_vector = self.vectorizer.vectorize(sample["string"])
    #         output_vector = self.classifier.predict(input_vector)
    #         predicted.append(self.vectorizer.decode_labels(output_vector))
    #         labeled.append(sample["label"])

    #     macro_f1 = custom_macro_f1_score(predicted, labeled)
    #     micro_f1 = custom_micro_f1_score(predicted, labeled)

    #     if self.params.log_metrics:
    #         mlflow.log_metric("Test Macro F1", macro_f1)
    #         mlflow.log_metric("Test Micro F1", micro_f1)

    #     return macro_f1, micro_f1


# class Predictor:
#     def __init__(self):
#         # find the components of the trainer:
#         vectorizer_section = config[config["predictor"]["vectorizer"]]
#         classifier_section = config[config["predictor"]["classifier"]]
#         # find what to instantiate:
#         vectorizer_class = utils.import_module(vectorizer_section["module"])
#         classifier_class = utils.import_module(classifier_section["module"])
#         # instantiate objects of the wished components:
#         self.vectorizer = vectorizer_class(vectorizer_section)
#         self.classifier = classifier_class(classifier_section)

#     def predict(self, data):
#         input_vector = self.vectorizer.vectorize(data)
#         output_vector = self.classifier.predict(input_vector)
#         return self.vectorizer.decode_labels(output_vector)


def run(args, config):
    if args.train:
        trainer = Trainer(args, config)
        if "multitask_trainer" in config:
            print("Running multitask trainer...")
            trainer.keras_multitask(args)
            tf.keras.backend.clear_session()
        if "singletask_trainer" in config:
            print("Running singletask trainer...")
            trainer.keras_singletask(args)


if __name__ == "__main__":
    # reader = SciciteReader(config["trainer"]["dataset"])
    # text, labels, sections, worthiness = reader.load_multitask_data()
    # # print(label_encoder.texts_to_sequences(['background True is background background false sometimes']))
    # keras_model = MultitaskLearner(
    #     config["multitask_trainer"]
    # )
    # text_tensor, vocab = keras_model.prepare_data(text, typ="text", pad_token="<PAD>")  # <PAD> specific for elmo
    # labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)
    # sections_tensor, sections_tokenizer = keras_model.prepare_data(sections)
    # worthiness_tensor, worthiness_tokenizer = keras_model.prepare_data(worthiness)

    # dataset = keras_model.create_dataset(
    #     text_tensor,
    #     labels_tensor,
    #     sections_tensor,
    #     worthiness_tensor
    # )

    # # example_input_batch, example_labes, _, _ = next(iter(dataset))
    # # for element in dataset.as_numpy_iterator():
    # #     print(element)
    # #     print('########')
    # vocab_size = len(vocab)
    # labels_size = len(labels_tokenizer.word_index.keys())
    # section_size = len(sections_tokenizer.word_index.keys())
    # worthiness_size = len(worthiness_tokenizer.word_index.keys())

    # keras_model.create_model(
    #     vocab_size, labels_size, section_size, worthiness_size
    # )
    # keras_model.fit_model(dataset)

    # _______________________

    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        default=False,
        help="Set to True if training, else set to False.",
    )
    parser.add_argument(
        "--test",
        required=False,
        default=True,
        help="Set to True if testing, else set to False.",
    )
    parser.add_argument(
        "--log_metrics",
        required=False,
        default=False,
        help="Set to True if metrics should me logged with mlflow, else set to False.",
    )
    parser.add_argument(
        "--config",
        required=False,
        default="configs/keras/default.conf",
        help="Path to configuration file.",
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print("Loading configuration from", args.config)

    run(args, config)  # main function

    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time:", str(datetime.timedelta(seconds=total_time)))