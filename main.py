from src.utils.utils import BOW
from src.utils.reader import SciciteReader
from src.utils.reader import SciciteReader
from src.models.perceptron import Perceptron

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
    # prepare data
    reader = SciciteReader(dataset)
    train_set, dev_set, test_set = reader.load_tdt()
    # init feature extractor
    bow = BOW(stopwords_path="resources/stopwords_en.txt")
    # init classifier
    dim = bow.get_dimensionality()
    model = Perceptron(0.01, 5, dim)

    # build training set
    text = [sample["string"] for sample in train_set]
    train_set_inputs = bow.generate(text, min_occurrence=30)
    train_set_labels = [sample["label"] for sample in train_set]
    assert len(train_set_inputs) == len(train_set_labels)


if __name__ == "__main__":
    # init trainer
    train("data/scicite/")
