import os
import json


class SciciteReader:
    """Load Scicite data

    This implementation loads the Scicite dataset.

    Parameters
    ----------
    data_dir: string, absolute path to the data directory of the Scicite dataset

    Examples
    --------
    >>> from reader import SciciteReader
    >>> reader = SciciteReader("path/to/data")
    >>> train, dev, test = reader.load_tdt()
    >>> print("Train:", len(train), "Dev:", len(dev), "Test", len(test))
    ... Train: 8243 Dev: 916 Test: 1861
    >>> scaffold_cite = reader.load_scaffold("cite")
    >>> scaffold_cite_sentences = read_sentences()
    >>> print(len(scaffold_cite_sentences))
    ... 73484
    >>> scaffold_title = reader.load_scaffold("title")
    >>> scaffold_title_sentences = read_sentences()
    >>> print(len(scaffold_title_sentences))
    ... 91412
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_tdt(self):
        """
        Loads train, dev, and test data.
        :return: list, a list with the format [train, dev, test]
        """
        train_file = os.path.join(self.data_dir, "train.jsonl")
        dev_file = os.path.join(self.data_dir, "dev.jsonl")
        test_file = os.path.join(self.data_dir, "test.jsonl")

        tdt = []

        for file in [train_file, dev_file, test_file]:
            with open(file, "r") as f:
                data = [json.loads(x) for x in list(f)]
                tdt.append(data)
        return tdt

    def load_scaffold(self, scaffold):
        """
        Loads scaffold data.
        :param scaffold: string, choose from {'cite', 'title'}
        :return: list
        """
        paths = {"cite": os.path.join(self.data_dir, "scaffolds", "cite-worthiness-scaffold-train.jsonl"),
                 "title": os.path.join(self.data_dir, "scaffolds", "sections-scaffold-train.jsonl")}

        file = paths[scaffold]
        with open(file, "r") as f:
            data = [json.loads(x) for x in list(f)]
        return data

    @staticmethod
    def read_sentences(data):
        """
        Extracts sentences from a list of dicts.
        :param data: list, a dataset in the Scicite format
        :return: list, a list of strings (each is a sentence)
        """
        k = "text" if "text" in data[0] else "string"
        return [d[k] for d in data]
