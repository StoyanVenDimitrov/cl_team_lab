"""Data readers"""
import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf


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
        self.vocab_set = set()
        self.section_set = set()
        self.worthiness_set = set()
        self.labels_set = set()
        self.tokenizer = tfds.features.text.Tokenizer()

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
            with open(file, "r") as data_file:
                data = [json.loads(x) for x in list(data_file)]
                tdt.append(data)
        return tdt

    def load_main_task_data(self, dev=False):
        """Loads train data for main task"""
        if dev:
            file = os.path.join(self.data_dir, "dev.jsonl")
        else:
            file = os.path.join(self.data_dir, "train.jsonl")

        with open(file, "r") as data_file:
            data = [json.loads(x) for x in list(data_file)]
        for sample in data:
            tokens = self.tokenizer.tokenize(sample["string"])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
            #self.labels_set.update([sample["label"]])
            sample["relevant_key"] = "label"
            # sample["text"] = [sample["string"]]
        return data

    def load_scaffold(self, scaffold):
        """
        Loads scaffold data.
        :param scaffold: string, choose from {'cite', 'title'}
        :return: list
        """
        paths = {
            "cite": os.path.join(
                self.data_dir, "scaffolds", "cite-worthiness-scaffold-train.jsonl"
            ),
            "title": os.path.join(
                self.data_dir, "scaffolds", "sections-scaffold-train.jsonl"
            ),
        }

        file = paths[scaffold]
        with open(file, "r") as scaffold_file:
            data = [json.loads(x) for x in list(scaffold_file)]
        for sample in data:
            # sample["text"] = [sample["text"]]
            if scaffold == "cite":
                # self.worthiness_set.update([str(sample["is_citation"])])
                sample["relevant_key"] = "is_citation"
                sample['is_citation'] = str(sample["is_citation"])
            else:
                sample["relevant_key"] = "section_title"
            tokens = self.tokenizer.tokenize(sample["text"])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
        return data

    def load_multitask_data(self, for_validation=False, multitask=False):

        data = []

        not_encoded_tokens = []
        text = []
        labels = []
        sections = []
        worthiness = []

        if multitask:
            for i, j, k in zip(
                self.load_main_task_data(),
                self.load_scaffold("cite"),
                self.load_scaffold("title"),
            ):
                data.append(i)
                data.append(j)
                data.append(k)
        else:
            for i in self.load_main_task_data(dev=for_validation):
                data.append(i)


        for sample in data[:20]:
            if "text" in sample.keys():
                text.append(sample['text'])
            elif "string" in sample.keys():
                text.append(sample['string'])
            not_encoded_tokens.append(sample["tokens"])

            relevant_key = sample["relevant_key"]
            
            if relevant_key == "label":
                    labels.append(sample["label"])
            else:
                labels.append("__unknown__")

            if multitask:
                if relevant_key == "section_title":
                    # avoid 'related work' to be tokenized with two labels
                    sections.append(sample["section_title"].split(' ')[0])
                else:
                    sections.append("__unknown__")

                if relevant_key == "is_citation":
                    worthiness.append(sample["is_citation"])
                else:
                    worthiness.append("__unknown__")

        return text, labels, sections, worthiness

    @staticmethod
    def read_sentences(data):
        """
        Extracts sentences from a list of dicts.
        :param data: list, a dataset in the Scicite format
        :return: list, a list of strings (each is a sentence)
        """
        k = "text" if "text" in data[0] else "string"
        return [d[k] for d in data]

    def get_dimensions(self):
        return len(self.vocab_set), len(self.labels_set), len(self.section_set), len(self.worthiness_set)
