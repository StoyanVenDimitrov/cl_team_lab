"""
Simple perceptron algorithm
"""
import pickle
import os
from src.models.model import Model


class Perceptron(Model):
    """
    Implementation from scratch
    """

    def __init__(self, config):
        super().__init__(config)
        self.weights = []
        # TODO: must be in the config file
        # self.model_path = "saved_models/classifiers/"
        self.load_model(self.model_path)

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self, training_inputs):
        print('+++ Training a new model for ', self.__class__.__name__, "+++")
        # dimensions: len of vocabulary x len of labels()
        dimensionality = training_inputs[0]
        # each row represents one perceptron; first weights dimension is bias
        self.weights = [
            [0.0 for _ in range(len(dimensionality[0]) + 1)]
            for _ in range(len(dimensionality[1]))
        ]
        for _ in range(self.number_of_epochs):
            for row in training_inputs:
                self.weight_update(row)
        self.save_model()
        return self.weights

    def predict(self, sent_representation):
        score_list = []
        for weights in self.weights:
            activation = self.predict_binary(sent_representation, weights)
            score_list.append(activation)
        argmax = score_list.index(max(score_list))
        labels = [0 for _ in score_list]
        labels[argmax] = 1
        return labels

    def save_model(self, model=None):
        """simply save the weights"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filename = os.path.join(
            self.model_path, "{}.pickle".format(self.__class__.__name__)
        )
        pickle.dump(self.weights, open(filename, "wb"))
        print("Model saved for ", self.__class__.__name__)

    def load_model(self, path):
        try:
            filename = os.path.join(
                self.model_path, "{}.pickle".format(self.__class__.__name__)
            )
            self.weights = pickle.load(open(filename, "rb"))
            print("Model loaded for ", self.__class__.__name__)
        except FileNotFoundError:
            print("You start with a fresh model for ", self.__class__.__name__)

    def weight_update(self, row):
        """
        do the update one sample at a time
        :param row: training sample
        :return:
        """
        prediction = self.predict(row[0])
        labels = row[1]
        for i in range(len(row[1])):
            error = labels[i] - prediction[i]
            # decreasing if wrong because error = -1 and increasing for error = 1,
            # otherwise unchanged
            self.weights[i][0] += self.learning_rate * error
            for j in range(len(row[0])):
                self.weights[i][j + 1] = (
                    self.weights[i][j + 1] + self.learning_rate * error * row[0][j]
                )

    # Make a BINARY prediction with weights
    @staticmethod
    def predict_binary(sent_representation, weights):
        """
        get activation value from set of weights
        :param weights: weights for this perceptron
        :param sent_representation: vector representation of a sentence
        :return: class
        """
        activation = weights[0]  # the bias
        for i, _ in enumerate(sent_representation):
            activation += weights[i + 1] * sent_representation[i]
        return activation
