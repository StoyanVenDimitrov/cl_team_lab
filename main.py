from src.utils.utils import load_config
from src.utils.bow import BOW
from src.utils.utils import make_filename
from src.utils.reader import SciciteReader
from src.utils.reader import SciciteReader
from src.models.perceptron import Perceptron

import os
import argparse
import mlflow

# pylint:skip-file


class Trainer:
    def __init__(self, bow, model):
        self.bow = bow
        self.model = model

    def vectorize_data(self, data):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass


def train(dataset):
    # TODO: replace this function with Trainer object
    """
    train with dataset
    :return:
    """
    root = os.path.join(os.path.abspath(__file__), os.pardir, os.pardir)
    # load config
    config_file = os.path.join(root, "configs", "default.conf")
    config = load_config(config_file)

    # prepare data
    reader = SciciteReader(dataset)
    train_set, dev_set, test_set = reader.load_tdt()
    # init feature extractor
    bow = BOW(stopwords_path="resources/stopwords_en.txt")

    # build vectorized training set
    text = [sample["string"] for sample in train_set]

    # generate or load bow model
    filename = make_filename(config["features"])
    path = os.path.join(root, "saved_models", "features", config["features"]["model"], filename)
    if os.path.isfile(path):
        bow = bow.load_model(path)
    else:
        bow.generate(text, config=config, filename=path)
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
    model = Perceptron(1, 5, dim)

    # predict
    vector_input = bow.vectorize(text[0])
    prediction_vector = model.predict(vector_input)
    print(bow.decode_labels(all_labels, prediction_vector))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", required=True, default=False, help="Set to True if training, else set to False.")
    parser.add_argument("--do_test", required=False, default=False, help="Set to True if testing, else set to False.")

    # init trainer
    train("data/scicite/")

    # TODO FLAGS
