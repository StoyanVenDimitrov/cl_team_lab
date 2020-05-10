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

    # build vectorized training set
    text = [sample["string"] for sample in train_set]
    bow.generate(text, min_occurrence=200)
    train_set_labels = [sample["label"] for sample in train_set]
    all_labels = set(train_set_labels)

    train_set_inputs = [
        (
            bow.vectorize(sample["string"]),
            bow.vectorize_labels(all_labels, sample["label"]),
        )
        for sample in train_set
    ]

    # init classifier
    dim = (bow.get_dimensionality(), len(set(train_set_labels)))
    model = Perceptron(1, 5, dim)

    # predict
    vector_input = bow.vectorize(text[0])
    prediction_vector = model.predict(vector_input)
    print(bow.decode_labels(all_labels, prediction_vector))


if __name__ == "__main__":
    # init trainer
    train("data/scicite/")
