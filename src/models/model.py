"""models Model functionalities"""
import abc


class Model:
    """Abstract class for classifiers"""

    def __init__(self, config):
        self.learning_rate = int(config["learning_rate"])
        self.number_of_epochs = int(config["number_of_epochs"])
        self.model_path = config["model_path"]

    @abc.abstractmethod
    def train(self, training_inputs):
        """
        do the training on train set
        :param training_inputs: list of (feature vector, label) from input set
        :return: weights
        """
        return

    @abc.abstractmethod
    def predict(self, sent_representation):
        """
        predict class of a sentence
        :param sent_representation: vector representation of a sentence
        :return: class
        """
        return

    @abc.abstractmethod
    def save_model(self, model=None):
        """
        save the model to reuse it for inference
        :return:
        """
        return

    @abc.abstractmethod
    def load_model(self, path):
        """
        load saved model
        :return: model
        """
        return
