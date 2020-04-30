from collections import defaultdict
import os


class BOW:
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
    >>> from utils import BOW
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
        input_type="sentences",
        stopwords_lang="en",
        stopwords_path=None,
        lowercase=True,
    ):
        self.input_type = input_type
        self.stopwords_lang = stopwords_lang
        self.stopwords_path = stopwords_path
        self.lowercase = lowercase

        stopwords_langs = ["en"]

        if self.stopwords_lang in stopwords_langs and not self.stopwords_path:
            self.stopwords_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                "data",
                f"stopwords_{self.stopwords_lang}.txt",
            )

        if self.stopwords_path:
            with open(self.stopwords_path, "r") as f:
                self.stopwords = f.read().split("\n")

    def split_into_sentences(self, text):
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

    def get_word_frequencies(self, tokens, sort=False, descending=True):
        """
        Counts word frequencies in a text.
        :param tokens: list, this should be a list of strings (each being a single token)
        :param sort: bool, set to True if the output list should be sorted
        :param descending: bool, set to True if the output list should be sorted in a descending order
        :return: dict, a list of tuples made up of (string, int), the string is the word and the integer its frequency
        """
        d = defaultdict(int)

        for t in tokens:
            d[t] += 1

        if sort:
            d = sorted(d.items(), key=lambda k: k[1], reverse=descending)
        else:
            d = list(d.items())

        return d

    def generate(self, text, top_n=None, min_occurrence=None, sort=True):
        """
        Function to generate bag-of-words representation of text.
        :param text: string, this should be the entire text in a document
        :param min_occurrence: int, minimum number of occurrences for words
        :param sort: bool, parameter for sorting word frequency
        :return: list: list of lists containing integers
        """
        if self.input_type == "sentences":
            sentences = [t.lower() for t in text]
        elif self.input_type == "text":
            if self.stopwords_path:
                tokens = self.remove_stopwords(text)

            sentences = self.split_into_sentences(text)

        if self.stopwords_path and self.input_type == "sentences":
            tokens = self.remove_stopwords(" ".join(sentences))
            tokens = tokens.split(" ")
        else:
            tokens = " ".join(sentences).split(" ")

        word_frequencies = self.get_word_frequencies(tokens, sort=sort)

        if min_occurrence:
            word_frequencies = [x for x in word_frequencies if x[1] > min_occurrence]

        # pick the top n
        if top_n:
            words = [w for w, _ in word_frequencies[:top_n]]
        else:
            words = [w for w, _ in word_frequencies]

        # generate bag-of-words
        bow = []
        for sent in sentences:
            row = [0] * len(words)
            for i, w in enumerate(sent.lower().split(" ")):
                idx = words.index(w) if w in words else None
                if idx is not None:
                    row[idx] += 1
            bow.append(row)

        return bow
