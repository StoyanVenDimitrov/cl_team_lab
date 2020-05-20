"""
Simple perceptron algorithm
"""
import pickle
import os
from src.models.model import Model
from src import evaluation
import mlflow
from tqdm import tqdm



class Perceptron(Model):
    """
    Implementation from scratch
    """

    def __init__(self, config, params):
        super().__init__(config)
        self.params = params
        self.weights = []
        # TODO: must be in the config file
        # self.model_path = "saved_models/classifiers/"
        self.load_model(self.model_path)

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self, training_inputs, dev_inputs):
        print("+++ Training a new model for ", self.__class__.__name__, "+++")
        # dimensions: len of vocabulary x len of labels()
        dimensionality = training_inputs[0]
        # each row represents one perceptron; first weights dimension is bias
        self.weights = [
            [0.0 for _ in range(len(dimensionality[0]) + 1)]
            for _ in range(len(dimensionality[1]))
        ]
        for _ in tqdm(range(self.number_of_epochs)):
            for row in training_inputs[:300]:
                self.weight_update(row)
            # evaluate after each epoch:
            micro_f1, macro_f1 = self.evaluate_on_dev_set(dev_inputs)
            self.statistics['micro_f1'].append(micro_f1)
            self.statistics['macro_f1'].append(macro_f1)
        self.save_model()
        return self.weights

    def predict(self, sent_representation):
        try:
            score_list = []
            for weights in self.weights:
                activation = self.predict_binary(sent_representation, weights)
                score_list.append(activation)
            argmax = score_list.index(max(score_list))
            labels = [0 for _ in score_list]
            labels[argmax] = 1
            return labels
        except ValueError:
            raise ValueError("Cannot predict, model not found!")

    def evaluate_on_dev_set(self, dev_set):
        """
        evaluate the performance on dev set
        :param dev_set:
        :return:
        """
        predictions = []
        true_labels = []
        for sample in dev_set:
            labels = self.predict(sample[0])
            # take the indices of the predicted label instead of decoding it
            predictions.append(labels.index(1))
            true_labels.append(sample[1].index(1))
        macro_f1 = evaluation.custom_macro_f1_score(predictions, true_labels)
        micro_f1 = evaluation.custom_micro_f1_score(predictions, true_labels)

        return micro_f1, macro_f1

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
            if self.params.log_metrics:
                mlflow.log_metric("Loss", error)
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
