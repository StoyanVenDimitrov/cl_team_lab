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
        # each row represents one perceptron; first weights dimension is bias
        # VERY F***ING IMPORTANT:
        # self.weights = [0.0 for _ in range(dimensionality[0] + 1)]*dimensionality[1]
        # is not the same as:
        self.weights = [
            [0.0 for _ in range(dimensionality[0] + 1)]
            for _ in range(dimensionality[1])
        ]

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self, training_inputs):
        """
        do the training on train set
        :param training_inputs: list of (feature vector, label) from input set
        :return: weights
        """
        for _ in range(self.number_of_epochs):
            for row in training_inputs:
                self.weight_update(row)

        return self.weights

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

    def predict(self, sent_representation):
        """
        predict on multi-class
        :param sent_representation:
        :return: list: 1 on position of class, else 0s
        """
        score_list = []
        for weights in self.weights:
            activation = self.predict_binary(sent_representation, weights)
            score_list.append(activation)
        argmax = score_list.index(max(score_list))
        labels = [0 for _ in score_list]
        labels[argmax] = 1
        return labels

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
