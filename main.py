from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
from src.models.keras_model import MultitaskLearner
from src.models.keras_model import SingletaskLearner
import configparser
import argparse
import mlflow

# pylint:skip-file

config = configparser.ConfigParser()
config.read("configs/default.conf")


def keras_multitask():
    reader = SciciteReader(config["trainer"]["dataset"])
    text, labels, sections, worthiness = reader.load_data(multitask=True)

    text_dev, labels_dev, _, _ = reader.load_data(multitask=False, for_validation=True)
    keras_model = MultitaskLearner(
        config["multitask_trainer"]
    )
    text_tensor, text_tokenizer = keras_model.prepare_data(text)
    labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)
    sections_tensor, sections_tokenizer = keras_model.prepare_data(sections)
    worthiness_tensor, worthiness_tokenizer = keras_model.prepare_data(worthiness)

    text_tensor_dev = keras_model.prepare_dev_data(text_dev, text_tokenizer)
    labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)

    dataset = keras_model.create_dataset(
        text_tensor,
        labels_tensor,
        sections_tensor,
        worthiness_tensor
    )
    dev_dataset = keras_model.create_dev_dataset(
        text_tensor_dev,
        labels_tensor_dev
    )

    vocab_size = len(text_tokenizer.word_index.keys())
    labels_size = len(labels_tokenizer.word_index.keys())
    section_size = len(sections_tokenizer.word_index.keys())
    worthiness_size = len(worthiness_tokenizer.word_index.keys())

    keras_model.create_model(
        vocab_size, labels_size, section_size, worthiness_size
    )
    keras_model.fit_model(dataset, dev_dataset)


def keras_singletask():
    reader = SciciteReader(config["trainer"]["dataset"])
    text, labels, _, _ = reader.load_data(multitask=False)

    text_dev, labels_dev, _, _ = reader.load_data(multitask=False, for_validation=True)
    keras_model = SingletaskLearner(
        config["singletask_trainer"]
    )
    text_tensor, text_tokenizer = keras_model.prepare_data(text)
    labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)

    text_tensor_dev = keras_model.prepare_dev_data(text_dev, text_tokenizer)
    labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)

    dataset = keras_model.create_dataset(
        text_tensor,
        labels_tensor
    )
    dev_dataset = keras_model.create_dev_dataset(
        text_tensor_dev,
        labels_tensor_dev
    )

    vocab_size = len(text_tokenizer.word_index.keys())
    labels_size = len(labels_tokenizer.word_index.keys())

    keras_model.create_model(
        vocab_size, labels_size
    )
    keras_model.fit_model(dataset, dev_dataset)


if __name__ == "__main__":
    keras_multitask()
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "--train",
    #     required=True,
    #     default=False,
    #     help="Set to True if training, else set to False.",
    # )
    # parser.add_argument(
    #     "--test",
    #     required=False,
    #     default=False,
    #     help="Set to True if testing, else set to False.",
    # )
    # parser.add_argument(
    #     "--log_metrics",
    #     required=False,
    #     default=False,
    #     help="Set to True if metrics should me logged with mlflow, else set to False.",
    # )
    # parser.add_argument(
    #     "--train_features",
    #     required=False,
    #     default=True,
    #     help="Set to True if training features, else set to False.",
    # )
    #
    # args = parser.parse_args()
    #
    # if args.log_metrics:
    #     mlflow.start_run()
    #     mlflow.log_params(utils.get_log_params(config))
    #
    # trainer = Trainer(args)
    # if args.train_features:
    #     trainer.train_feature_extractor()
    # if args.train:
    #     trainer.train_classifier()
    # macro_f1, micro_f1 = trainer.evaluate()
    # print(f"Macro F1: {macro_f1}\nMicro F1: {micro_f1}")
    #
    # # if args.test:
    # # predictor = Predictor()
    # # print(predictor.predict("Set to True if testing, else set to False."))
    #
    # if args.log_metrics:
    #     mlflow.end_run()
