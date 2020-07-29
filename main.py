from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
from src.models.keras_model import MultitaskLearner
from src.models.keras_model import SingletaskLearner
import configparser
import argparse
import time
import datetime
import tensorflow as tf

# pylint:skip-file

config = configparser.ConfigParser()
config.read("configs/default.conf")


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def train_multitask(self):
        start_time = time.time()

        max_len = int(self.config["multitask_trainer"]["max_len"])

        reader = SciciteReader(self.config["preprocessor"])
        print("Loading data...")
        text, labels, sections, worthiness = reader.load_data(_type="train", multitask=True)
        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        text_test, labels_test, _, _ = reader.load_data(_type="test", multitask=False)
        
        # reader = SciciteReader(config["trainer"]["dataset"])
        # text, labels, sections, worthiness = reader.load_data(multitask=True)

        # text_dev, labels_dev, _, _ = reader.load_data(multitask=False, for_validation=True)
        keras_model = MultitaskLearner(
            self.config
        )
        
        input_ids, input_masks, input_segments = keras_model.prepare_input_data(text)
        dev_input_ids, dev_input_masks, dev_input_segments = keras_model.prepare_input_data(text_dev)
        test_input_ids, test_input_masks, test_input_segments = keras_model.prepare_input_data(text_test)

        print("Preparing data...")
        labels_tensor, labels_tokenizer = keras_model.prepare_output_data(labels)
        sections_tensor, sections_tokenizer = keras_model.prepare_output_data(sections)
        worthiness_tensor, worthiness_tokenizer = keras_model.prepare_output_data(worthiness)

        dev_label_tensor = keras_model.prepare_dev_output_data(labels_dev, labels_tokenizer)
        test_label_tensor = keras_model.prepare_dev_output_data(labels_test, labels_tokenizer)

        dataset = keras_model.create_dataset(
            input_ids,
            input_masks,
            input_segments,
            labels_tensor,
            sections_tensor,
            worthiness_tensor
        )
        dev_dataset = keras_model.create_dev_dataset(
            dev_input_ids, 
            dev_input_masks, 
            dev_input_segments,
            dev_label_tensor
        )
        test_dataset = keras_model.create_dev_dataset(
            test_input_ids, 
            test_input_masks, 
            test_input_segments,
            test_label_tensor
        )

        # example_input_batch, example_labes, _, _ = next(iter(dataset))
        # for element in dataset.as_numpy_iterator():
        #     print(element)
        #     print('########')
        #vocab_size = len(text_tokenizer.word_index.keys())
        labels_size = len(labels_tokenizer.word_index.keys())
        section_size = len(sections_tokenizer.word_index.keys())
        worthiness_size = len(worthiness_tokenizer.word_index.keys())

        print("Creating model...")
        keras_model.create_model(
            labels_size, section_size, worthiness_size
        )
        print("Fitting model...")
        keras_model.fit_model(dataset, dev_dataset)

        print("Saving model...")
        keras_model.save_model()

        print("Evaluating...")
        keras_model.eval(test_dataset, save_output=True)

        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))


    def train_singletask(self):
        start_time = time.time()

        max_len = int(self.config["singletask_trainer"]["max_len"])

        reader = SciciteReader(self.config["preprocessor"])
        print("Loading data...")
        text, labels, sections, worthiness = reader.load_data(_type="train", multitask=True)
        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        text_test, labels_test, _, _ = reader.load_data(_type="test", multitask=False)
        
        # reader = SciciteReader(config["trainer"]["dataset"])
        # text, labels, sections, worthiness = reader.load_data(multitask=True)

        # text_dev, labels_dev, _, _ = reader.load_data(multitask=False, for_validation=True)
        keras_model = SingletaskLearner(
            self.config
        )
        
        input_ids, input_masks, input_segments = keras_model.prepare_input_data(text)
        dev_input_ids, dev_input_masks, dev_input_segments = keras_model.prepare_input_data(text_dev)
        test_input_ids, test_input_masks, test_input_segments = keras_model.prepare_input_data(text_test)

        print("Preparing data...")
        labels_tensor, labels_tokenizer = keras_model.prepare_output_data(labels)

        dev_label_tensor = keras_model.prepare_dev_output_data(labels_dev, labels_tokenizer)
        test_label_tensor = keras_model.prepare_dev_output_data(labels_test, labels_tokenizer)

        dataset = keras_model.create_dataset(
            input_ids,
            input_masks,
            input_segments,
            labels_tensor,
        )
        dev_dataset = keras_model.create_dev_dataset(
            dev_input_ids, 
            dev_input_masks, 
            dev_input_segments,
            dev_label_tensor
        )
        test_dataset = keras_model.create_dev_dataset(
            test_input_ids, 
            test_input_masks, 
            test_input_segments,
            test_label_tensor
        )

        # example_input_batch, example_labes, _, _ = next(iter(dataset))
        # for element in dataset.as_numpy_iterator():
        #     print(element)
        #     print('########')
        #vocab_size = len(text_tokenizer.word_index.keys())
        labels_size = len(labels_tokenizer.word_index.keys())

        print("Creating model...")
        keras_model.create_model(labels_size)

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
        # if "multitask_trainer" in config:
            # print("Running multitask trainer...")
            # trainer.train_multitask()
            # tf.keras.backend.clear_session()
        if "singletask_trainer" in config:
            print("Running singletask trainer...")
            trainer.train_singletask()


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
