from src.utils import utils
from src.evaluation import custom_macro_f1_score
from src.utils.reader import SciciteReader
import configparser
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

    def train_feature_extractor(self, training_data):
        self.vectorizer.generate(training_data)

    def train_classifier(self, training_data):
        train_set_inputs = [
            (
                self.vectorizer.vectorize(sample["string"]),
                self.vectorizer.vectorize_labels(sample["label"]),
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
        predicted, labeled = [], []
        for sample in self.test_set:
            input_vector = self.vectorizer.vectorize(sample['string'])
            output_vector = self.classifier.predict(input_vector)
            predicted.append(self.vectorizer.decode_labels(output_vector))
            labeled.append(sample['label'])

        return custom_macro_f1_score(predicted, labeled)


class Predictor:
    def __init__(self):
        # find the components of the trainer:
        vectorizer_section = config[config['predictor']['vectorizer']]
        classifier_section = config[config['predictor']['classifier']]
        # find what to instantiate:
        vectorizer_class = utils.import_module(vectorizer_section['module'])
        classifier_class = utils.import_module(classifier_section['module'])
        # instantiate objects of the wished components:
        self.vectorizer = vectorizer_class(vectorizer_section)
        self.classifier = classifier_class(classifier_section)

    def predict(self, data):
        input_vector = self.vectorizer.vectorize(data)
        output_vector = self.classifier.predict(input_vector)
        return self.vectorizer.decode_labels(output_vector)


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
    # trainer.train_feature_extractor(train_set)
    print(trainer.evaluate())

    # predictor = Predictor()
    # print(predictor.predict("Set to True if testing, else set to False."))
    # TODO FLAGS
