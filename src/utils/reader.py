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

    def load_main_task_data(self):
        """Loads train data for main task"""
        train_file = os.path.join(self.data_dir, "train.jsonl")

        with open(train_file, "r") as data_file:
            data = [json.loads(x) for x in list(data_file)]
        for sample in data:
            tokens = self.tokenizer.tokenize(sample["string"])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
            self.labels_set.update([sample["label"]])
            sample["relevant_key"] = "label"
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
            if scaffold == "cite":
                self.worthiness_set.update([str(sample["is_citation"])])
                sample["relevant_key"] = "is_citation"
            else:
                self.section_set.update([sample["section_title"]])
                sample["relevant_key"] = "section_title"
            tokens = self.tokenizer.tokenize(sample["text"])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
        return data

    def load_multitask_data(self):

        data = []

        not_encoded_tokens = []
        encoded_text = []
        encoded_labels = []
        encoded_sections = []
        encoded_worthiness = []

        for i, j, k in zip(
            self.load_main_task_data(),
            self.load_scaffold("cite"),
            self.load_scaffold("title"),
        ):
            data.append(i)
            data.append(j)
            data.append(k)

        text_encoder = tfds.features.text.TokenTextEncoder(self.vocab_set)
        label_encoder = tfds.features.text.TokenTextEncoder(self.labels_set)
        worthiness_encoder = tfds.features.text.TokenTextEncoder(self.worthiness_set)
        section_encoder = tfds.features.text.TokenTextEncoder(self.section_set)

        for sample in data:
            if "text" in sample.keys():
                encoded_text.append(text_encoder.encode(sample["text"]))
            elif "string" in sample.keys():
                encoded_text.append(text_encoder.encode(sample["string"]))
            not_encoded_tokens.append(sample["tokens"])

            relevant_key = sample["relevant_key"]

            if relevant_key == "label":
                encoded_labels.append(label_encoder.encode(sample["label"]))
            else:
                encoded_labels.append([-1])

            if relevant_key == "section_title":
                encoded_sections.append(section_encoder.encode(sample["section_title"]))
            else:
                encoded_sections.append([-1])

            if relevant_key == "is_citation":
                encoded_worthiness.append(
                    worthiness_encoder.encode(str(sample["is_citation"]))
                )
            else:
                encoded_worthiness.append([-1])

        def gen_train_series():
            for t, tl, ts, tw in zip(
                    encoded_text,
                    encoded_labels,
                    encoded_sections,
                    encoded_worthiness
            ):
                yield t, {'dense': tl, 'dense_1': ts, 'dense_2': tw}

        series = tf.data.Dataset.from_generator(gen_train_series,
                                                output_types=(
                                                    tf.int32,
                                                    {
                                                        'dense': tf.int32,
                                                        'dense_1': tf.int32,
                                                        'dense_2': tf.int32
                                                    }
                                                    ),
                                                )
        return series, text_encoder, label_encoder, section_encoder, worthiness_encoder

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
