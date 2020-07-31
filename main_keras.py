from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
from src.models.keras_model import MultitaskLearner
from src.models.keras_model import SingletaskLearner
import tensorflow as tf
import configparser
import argparse
import time
import datetime

# pylint:skip-file


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

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

        if embedding_type == "bert" or embedding_type == "albert":
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
        elif embedding_type == "bert" or embedding_type == "albert":
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
        keras_model.eval(test_dataset, save_output=False)

        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))

    def keras_singletask(self, args):
        start_time = time.time()

        # if self.args.log_metrics:
        #     utils.wandb_init_logs(self.config["singletask_trainer"])

        embedding_type = self.config["singletask_trainer"]["embedding_type"]
        max_len = int(self.config["singletask_trainer"]["max_len"])

        reader = SciciteReader(self.config["preprocessor"])
        print("Loading data...")
        text, labels, _, _ = reader.load_data(_type="train", multitask=False)
        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        text_test, labels_test, _, _ = reader.load_data(_type="test", multitask=False)

        keras_model = SingletaskLearner(self.config)

        if embedding_type == "bert" or embedding_type == "albert":
           input_ids, input_masks, input_segments = keras_model.prepare_input_data(text)
           dev_input_ids, dev_input_masks, dev_input_segments = keras_model.prepare_input_data(text_dev)
           test_input_ids, test_input_masks, test_input_segments = keras_model.prepare_input_data(text_test)
        elif embedding_type == "elmo":
            dataset = keras_model.create_dataset(
                text_tensor, labels_tensor, sections_tensor, worthiness_tensor
            )
            dev_dataset = keras_model.create_dev_dataset(text_tensor_dev, labels_tensor_dev)
            test_dataset = keras_model.create_dev_dataset(
                text_tensor_test, labels_tensor_test
            )

        print("Preparing data...")
        text_tensor, text_tokenizer = keras_model.prepare_data(
            text, max_len=max_len,
        )
        labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)

        text_tensor_dev = keras_model.prepare_dev_data(
            text_dev,
            text_tokenizer,
            max_len=max_len,
        )
        labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)

        text_tensor_test = keras_model.prepare_dev_data(
            text_test,
            text_tokenizer,
            max_len=max_len,
        )
        labels_tensor_test = keras_model.prepare_dev_data(labels_test, labels_tokenizer)

        print("Creating datasets...")
        dataset = keras_model.create_dataset(text_tensor, labels_tensor)
        dev_dataset = keras_model.create_dev_dataset(text_tensor_dev, labels_tensor_dev)
        test_dataset = keras_model.create_dev_dataset(
            text_tensor_test, labels_tensor_test
        )

        if embedding_type == "lstm":
            dataset = keras_model.create_dataset(text=text_tensor, labels=labels_tensor)
            dev_dataset = keras_model.create_dev_dataset(text=text_tensor_dev, labels=labels_tensor_dev)
            test_dataset = keras_model.create_dev_dataset(text=text_tensor_test, labels=labels_tensor_test)
        elif embedding_type == "bert" or embedding_type == "albert":
            dataset = keras_model.create_dataset(
                text=text_tensor,
                labels=labels_tensor,
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

        print("Creating model...")
        keras_model.create_model(vocab_size, labels_size)
        print("Fitting model...")
        keras_model.fit_model(dataset, dev_dataset)

        print("Saving model...")
        keras_model.save_model()

        print("Evaluating...")
        keras_model.eval(test_dataset, save_output=True)

        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))


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
