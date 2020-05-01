from src.utils.utils import import BOW
from src.utils.reader import SciciteReader


class Trainer:
    def __init__(self, bow, model):
        self.bow = bow
        self.model = model

    def vectorize_data(self, data):
        vectorized = []

    def train(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    # load BOW

    # init trainer
    trainer = Trainer(bow, model)
