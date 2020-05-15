"""
Utils for preparing the training
"""

from collections import defaultdict
from collections import OrderedDict
import os
import pickle

from src.utils.utils import make_filename
from src.utils.feature_extractor import FeatureExtractorModule


# pylint:disable=bad-continuation, invalid-name
class BOW(FeatureExtractorModule):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a Bag-Of-Words models.

    Parameters
    ----------
    input_type: string {'sentences', 'text'}

    stopwords_lan: string {'en'}

    stopwords_path: string, absolute path to the stopwords text file

    lowercase: boolean, True by default
        Convert all characters to lowercase.

    Attributes
    ----------
    stopwords: list
        A list of stopwords in string format.

    Examples
    --------
    >>> from bow import BOW
    >>> text = [
    ...     "Joe waited for the train",
    ...     "The train was late",
    ...     "Mary and Samantha took the bus at the bus station"
    ... ]
    >>> BOW = BOW()
    >>> bow = BOW.generate(text)
    >>> import pprint
    >>> pprint.pprint(bow)
    [[1, 0, 1, 1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 2, 0, 0, 0, 1, 1, 1, 1]]
    """

    def __init__(
        self,
        config,
        input_type="sentences",
        stopwords_lang="en"
    ):
        super().__init__(config)
        self.input_type = input_type
        self.stopwords_lang = stopwords_lang
        self.stopwords_path = config['stopwords_path']
        self.lowercase = (config['lowercase'] if 'lowercase' in config else None)

        self.filename = os.path.join(self.model_path, "{}.pickle".format(make_filename(self.config)))
        self.vocabulary = []
        self.all_labels = OrderedDict()  # simply using set messes up the order by loading

        if os.path.isfile(self.filename):
            self.load_model(self.model_path)
        else:
            stopwords_langs = ["en"]

            if self.stopwords_lang in stopwords_langs and not self.stopwords_path:
                self.stopwords_path = os.path.join(
                    os.path.dirname(__file__),
                    os.pardir,
                    os.pardir,
                    "resources",
                    f"stopwords_{self.stopwords_lang}.txt",
                )

            if self.stopwords_path:
                with open(self.stopwords_path, "r") as f:
                    self.stopwords = f.read().split("\n")

    def generate(self, train_set):
        """
        Function to generate bag-of-words representation of text.
        :param train_set: training set of content-label pairs
        :return:
        """
        if os.path.isfile(self.filename):
            print('+++ A model with that configuration already exists for ', self.__class__.__name__, '+++')
            print('+++ Loaded model for ', self.__class__.__name__, '+++')
            return

        print('+++ Generating a new model for ', self.__class__.__name__,'+++')
        text = [sample["string"] for sample in train_set]
        train_set_labels = [sample["label"] for sample in train_set]
        self.all_labels = OrderedDict.fromkeys(train_set_labels)
        top_n = (
            float(self.config["top_n"])
            if "top_n" in self.config
            else None
        )
        min_occurrence = (
            int(self.config["min_occurrence"])
            if "min_occurrence" in self.config
            else None
        )
        sort = (
            self.config["sort"]
            if "sort" in self.config
            else None
        )

        if self.input_type == "sentences":
            sentences = [t.lower() if self.lowercase else t for t in text]
        elif self.input_type == "text":
            if self.stopwords_path:
                text = self.remove_stopwords(text)

            sentences = self.split_into_sentences(text)
        if self.stopwords_path and self.input_type == "sentences":
            tokens = self.remove_stopwords(" ".join(sentences))
            tokens = tokens.split(" ")
        else:
            tokens = " ".join(sentences).split(" ")
        word_frequencies = self.get_word_frequencies(tokens, sort=sort)

        if min_occurrence:
            word_frequencies = [
                x
                for x in word_frequencies
                if x[1] > min_occurrence
            ]

        # pick the top n
        if top_n:
            _n = int(len(word_frequencies) * top_n)
            self.vocabulary = [w for w, _ in word_frequencies[:_n]]
        else:
            self.vocabulary = [w for w, _ in word_frequencies]
        if self.model_path:
            self.save_model()

    def vectorize(self, sentence):
        """
        build BOW vector from sentence
        :param sentence: String
        :return: list containing integers
        """
        bow = []
        if self.vocabulary:
            bow = [0] * len(self.vocabulary)
            sentence = sentence.lower() if self.lowercase else sentence
            for _, w in enumerate(sentence.split(" ")):
                idx = self.vocabulary.index(w) if w in self.vocabulary else None
                if idx is not None:
                    bow[idx] += 1
            return bow
        print("Cannot build BOW vector, vocabulary is missing.")
        return bow

    def save_model(self, model=None):
        """

        :return:
        """
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        # filename = os.path.join(
        #     self.model_path, "{}.pickle".format(self.filename)
        # )
        pickle.dump({'model': self.vocabulary, 'label_set': self.all_labels}, open(self.filename, "wb"))
        print("Model saved for ", self.__class__.__name__)
        print("Model path:", self.filename)

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        try:
            # filename = os.path.join(
            #     self.model_path, "{}.pickle".format(self.__class__.__name__)
            # )
            dumped = pickle.load(open(self.filename, "rb"))
            self.vocabulary = dumped['model']
            self.all_labels = dumped['label_set']
            print("Model loaded for ", self.__class__.__name__)
        except FileNotFoundError:
            print("You start with a fresh model for ", self.__class__.__name__)

    @staticmethod
    def split_into_sentences(text):
        """
        Splits a text into sentences.
        :param text: string, this should be a text with multiple sentences
        :return: list, a list of strings (each is a sentence)
        """
        return text.split(". ")

    def remove_stopwords(self, text):
        """
        Removes stopwords from a text.
        :param text: string, this can be a sentence or a text
        :return: string, the input text without the stopwords
        """
        return " ".join([w for w in text.split(" ") if w not in self.stopwords])

    @staticmethod
    def get_word_frequencies(tokens, sort=False, descending=True):
        """
        Counts word frequencies in a text.
        :param tokens: list, this should be a list of strings (each being a single token)
        :param sort: bool, set to True if the output list should be sorted
        :param descending: bool, set to True if the output list
                    should be sorted in a descending order
        :return: dict, a list of tuples made up of (string, int),
                    the string is the word and the integer its frequency
        """
        d = defaultdict(int)

        for t in tokens:
            d[t] += 1

        if sort:
            d = sorted(d.items(), key=lambda k: k[1], reverse=descending)
        else:
            d = list(d.items())

        return d

    def vectorize_labels(self, label):
        """
        :param label: string
        :return: list, 1 on label position, else 0
        """
        vector = [1 if i == label else 0 for i in self.all_labels]
        return vector

    def decode_labels(self, encoding_vector):
        """
        :param label_set:
        :param encoding_vector: list: 1 on place of the label in label_set
        :return: string: label name
        """
        for i, label in enumerate(self.all_labels):
            if encoding_vector[i] == 1:
                return label
        return None
