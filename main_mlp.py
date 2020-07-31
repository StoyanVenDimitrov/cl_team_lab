from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
import configparser
import argparse
import mlflow
from sklearn.metrics import classification_report

# pylint:skip-file

config = configparser.ConfigParser()
config.read("configs/mlp/default.conf")


class Trainer:
    def __init__(self, params):
        self.params = params
        # find the components of the trainer:
        vectorizer_section = config[config["trainer"]["vectorizer"]]
        classifier_section = config[config["trainer"]["classifier"]]
        # find what to instantiate:
        vectorizer_class = utils.import_module(vectorizer_section["module"])
        classifier_class = utils.import_module(classifier_section["module"])
        # instantiate objects of the wished components:
        self.vectorizer = vectorizer_class(vectorizer_section)
        self.classifier = classifier_class(classifier_section, self.params)

        self.reader = SciciteReader(config["preprocessor"])
        self.train_set, self.dev_set, self.test_set = self.reader.load_tdt()

    def train_feature_extractor(self, training_data=None):
        if not training_data:
            training_data = self.train_set + self.dev_set + self.test_set
        self.vectorizer.generate(training_data)

    def train_classifier(self, training_data=None, dev_data=None):
        if not training_data:
            training_data = self.train_set

        if not dev_data:
            dev_data = self.dev_set

        train_set_inputs = [
            (
                self.vectorizer.vectorize(sample["string"]),
                self.vectorizer.vectorize_labels(sample["label"]),
            )
            for sample in training_data
        ]
        dev_set_inputs = [
            (
                self.vectorizer.vectorize(sample["string"]),
                self.vectorizer.vectorize_labels(sample["label"]),
            )
            for sample in dev_data
        ]
        self.classifier.train(train_set_inputs, dev_set_inputs)
        self.statistics = self.classifier.get_train_statistics()

        if self.params.log_metrics:
            for i in range(len(self.statistics["macro_f1"])):
                mlflow.log_metric("Dev Macro F1", self.statistics["macro_f1"][i])
                mlflow.log_metric("Dev Micro F1", self.statistics["micro_f1"][i])

    def evaluate(self):
        predicted, labeled = [], []
        for sample in self.test_set:
            input_vector = self.vectorizer.vectorize(sample["string"])
            output_vector = self.classifier.predict(input_vector)
            predicted.append(self.vectorizer.decode_labels(output_vector))
            labeled.append(sample["label"])

        macro_f1 = custom_macro_f1_score(predicted, labeled)
        micro_f1 = custom_micro_f1_score(predicted, labeled)

        print(classification_report(labeled, predicted, ["background", "method", "result"]))
        report = classification_report(labeled, predicted, ["background", "method", "result"], output_dict=True)
        print(report)

        if self.params.log_metrics:
            mlflow.log_metric("Test Macro F1", macro_f1)
            mlflow.log_metric("Test Micro F1", micro_f1)

        return macro_f1, micro_f1


class Predictor:
    def __init__(self):
        # find the components of the trainer:
        vectorizer_section = config[config["predictor"]["vectorizer"]]
        classifier_section = config[config["predictor"]["classifier"]]
        # find what to instantiate:
        vectorizer_class = utils.import_module(vectorizer_section["module"])
        classifier_class = utils.import_module(classifier_section["module"])
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
        "--train",
        required=True,
        default=False,
        help="Set to True if training, else set to False.",
    )
    parser.add_argument(
        "--test",
        required=False,
        default=False,
        help="Set to True if testing, else set to False.",
    )
    parser.add_argument(
        "--log_metrics",
        required=False,
        default=False,
        help="Set to True if metrics should me logged with mlflow, else set to False.",
    )
    parser.add_argument(
        "--train_features",
        required=False,
        default=True,
        help="Set to True if training features, else set to False.",
    )

    args = parser.parse_args()

    if args.log_metrics:
        mlflow.start_run()
        mlflow.log_params(utils.get_log_params(config))

    trainer = Trainer(args)
    if args.train_features:
        trainer.train_feature_extractor()
    if args.train:
        trainer.train_classifier()
    macro_f1, micro_f1 = trainer.evaluate()
    print(f"Macro F1: {macro_f1}\nMicro F1: {micro_f1}")

    # if args.test:
    # predictor = Predictor()
    # print(predictor.predict("Set to True if testing, else set to False."))

    if args.log_metrics:
        mlflow.end_run()
