"""models Model functionalities"""
import abc


class Model:
    """Abstract class for classifiers"""

    def __init__(self, l_rate, n_epochs):
        self.learning_rate = l_rate
        self.number_of_epochs = n_epochs

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
