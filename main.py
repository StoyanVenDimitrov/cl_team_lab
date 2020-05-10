from src.utils.utils import load_config
from src.utils.bow import BOW
from src.utils.utils import make_filename
from src.utils.reader import SciciteReader
from src.utils.reader import SciciteReader
from src.models.perceptron import Perceptron
import configparser
import importlib
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
        vectorizer_class = _import_module(vectorizer_section['module'])
        classifier_class = _import_module(classifier_section['module'])
        # instantiate objects of the wished components:
        self.vectorizer = vectorizer_class(vectorizer_section)
        self.classifier = classifier_class(classifier_section)

    def vectorize_data(self, data):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass


def _import_module(dotted_path):
    module_parts = dotted_path.split(".")
    module_path = ".".join(module_parts[:-1])
    try:
        module = importlib.import_module(module_path, package=__package__)
    except ModuleNotFoundError:
        module = importlib.import_module(module_path)
    return getattr(module, module_parts[-1])

def train(dataset):
    # TODO: replace this function with Trainer object
    """
    train with dataset
    :return:
    """

    # prepare data
    reader = SciciteReader(dataset)
    train_set, dev_set, test_set = reader.load_tdt()
    # build vectorized training set
    text = [sample["string"] for sample in train_set]

    # init feature extractor
    bow = BOW(config["features"])
    bow.generate(text)
    # generate or load bow model


    # bow.generate(text, min_occurrence=200)
    train_set_labels = [sample["label"] for sample in train_set]
    all_labels = set(train_set_labels)
    train_set_inputs = [
        (
            bow.vectorize(sample["string"]),
            bow.vectorize_labels(all_labels, sample["label"]),
        )
        for sample in train_set
    ]
    print('done')
    # uncomment the following code to start mlflow
    # with mlflow.start_run():
    # TODO: log config
    # TODO: move logging around training block
    #     mlflow.log_metric("loss", loss)
    #     p,r,f1 = get_metrics()
    #     mlflow.log_metric("P", p)
    #     mlflow.log_metric("R", r)
    #     mlflow.log_metric("F1", f1)

    # init classifier
    dim = (bow.get_dimensionality(), len(set(train_set_labels)))
    model = Perceptron(config['perceptron'], dim)

    model.train(train_set_inputs)

    # predict
    print(text[5691])
    vector_input = bow.vectorize(text[5691])
    prediction_vector = model.predict(vector_input)
    print(bow.decode_labels(all_labels, prediction_vector))


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
    trainer = Trainer()
    # init trainer
    train("data/scicite/")

    # TODO FLAGS
