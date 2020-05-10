from src.utils.utils import load_config
from src.utils.bow import BOW
from src.utils import utils
from src.utils.utils import make_filename
from src.utils.reader import SciciteReader
from src.utils.reader import SciciteReader
from src.models.perceptron import Perceptron
import configparser
import os
import argparse
import mlflow

# pylint:skip-file

config = configparser.ConfigParser()
config.read('configs/default.conf')


class Trainer:
    def __init__(self):
        # find the components of the trainer:
        vectorizer_section = config[config['trainer']['vectorizer']]
        classifier_section = config[config['trainer']['classifier']]
        # find what to instantiate:
        vectorizer_class = utils.import_module(vectorizer_section['module'])
        classifier_class = utils.import_module(classifier_section['module'])
        # instantiate objects of the wished components:
        self.vectorizer = vectorizer_class(vectorizer_section)
        self.classifier = classifier_class(classifier_section)

        reader = SciciteReader(config['trainer']['dataset'])
        self.train_set, self.dev_set, self.test_set = reader.load_tdt()

    def vectorize_data(self, data):
        return self.vectorizer.vectorize(data)

    def train(self, training_data):
        self.vectorizer.generate(training_data)
        train_set_labels = [sample["label"] for sample in training_data]
        all_labels = set(train_set_labels)
        train_set_inputs = [
            (
                self.vectorizer.vectorize(sample["string"]),
                self.vectorizer.vectorize_labels(all_labels, sample["label"]),
            )
            for sample in training_data
        ]
        self.classifier.train(train_set_inputs)

        # uncomment the following code to start mlflow
        # with mlflow.start_run():
        # TODO: log config
        # TODO: move logging around training block
        #     mlflow.log_metric("loss", loss)
        #     p,r,f1 = get_metrics()
        #     mlflow.log_metric("P", p)
        #     mlflow.log_metric("R", r)
        #     mlflow.log_metric("F1", f1)

    def evaluate(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--do_train",
        required=True,
        default=False,
        help="Set to True if training, else set to False.",
    )
    parser.add_argument(
        "--do_test",
        required=False,
        default=False,
        help="Set to True if testing, else set to False.",
    )

    reader = SciciteReader("data/scicite/")
    train_set, dev_set, test_set = reader.load_tdt()
    trainer = Trainer()
    # init trainer
    trainer.train(train_set)

    # TODO FLAGS
