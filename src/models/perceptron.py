"""
Simple perceptron algorithm
"""
from src.models.model import Model


class Perceptron(Model):
    """
    Implementation from scratch
    """

    def __init__(self, l_rate, n_epochs, dimensionality):
        super().__init__(l_rate, n_epochs)
        # first weights dimension is bias
        self.weights = [0.0 for _ in range(dimensionality + 1)]

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self, training_inputs, training_labels):
        """
        do the training on train set
        :param training_inputs: list of lists: feature vectors from input strings
        :param training_labels: list of strings: labels
        :return: weights
        """

        for _ in range(self.number_of_epochs):
            for row in training_inputs:
                prediction = self.predict(row)
                error = row[-1] - prediction
                self.weights[0] = (
                    self.weights[0] + self.learning_rate * error
                )  # the bias
                for i in range(len(row) - 1):
                    self.weights[i + 1] = (
                        self.weights[i + 1] + self.learning_rate * error * row[i]
                    )
        return self.weights

    # Make a BINARY prediction with weights
    def predict(self, sent_representation):
        """
        predict class of a sentence
        :param sent_representation: vector representation of a sentence
        :return: class
        """
        activation = self.weights[0]  # the bias
        for i in range(len(sent_representation) - 1):
            activation += self.weights[i + 1] * sent_representation[i]
        return 1.0 if activation >= 0.0 else 0.0
