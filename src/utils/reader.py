"""Data readers"""
import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import random
import spacy
import nlp
import pandas as pd
import jsonlines


class SciciteReaderNLP:
    def __init__(self, config):
        self.data_dir = config["dataset"]
        self.lemmatize = True if config["lemmatize"] == "True" else False
        self.do_balance_dataset = True if config["balance_dataset"] == "True" else False
        self.shuffle_data = True if config["shuffle_data"] == "True" else False

        if self.lemmatize:
            self.nlp = spacy.load("en_core_sci_lg")

    def load_dataset(self, file):
        json_data = []

        labels = {"method": 0, "background": 1, "result": 2}

        with jsonlines.open(file) as reader:
            for obj in reader:
                obj["label"] = labels[obj["label"]]
                json_data.append(obj)
            
        df = pd.DataFrame(json_data)

        return nlp.Dataset.from_pandas(df)
    
    def preprocess_data(self):
        print("Processing data...")

        train_file = os.path.join(self.data_dir, "train.jsonl")
        dev_file = os.path.join(self.data_dir, "dev.jsonl")
        test_file = os.path.join(self.data_dir, "test.jsonl")
        
        train = self.load_dataset(train_file)
        dev = self.load_dataset(dev_file)
        test = self.load_dataset(test_file)

        if self.do_balance_dataset:
            train, dev = self.balance_data(train), self.balance_data(dev)

        return train, dev, test
        
    def balance_data(self, dataset):
        print("Balancing data...")
        class0 = []
        class1 = []
        class2 = []

        for i, e in enumerate(dataset):
            if e["label"] == 0:
                class0.append(i)
            elif e["label"] == 1:
                class1.append(i)
            elif e["label"] == 2:
                class2.append(i)

        sample_size = min(len(class0), len(class1), len(class2))

        bclass0 = random.sample(class0, sample_size)
        bclass1 = random.sample(class1, sample_size)
        bclass2 = random.sample(class2, sample_size)

        bidxs = bclass0 + bclass1 + bclass2
        if self.shuffle_data:
            random.shuffle(bidxs)

        def keep(instance, idx):
            return idx in bidxs

        bdataset = dataset.filter(keep, with_indices=True, load_from_cache_file=False)

        return bdataset


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

    def __init__(self, config):
        self.config = config
        self.data_dir = config["dataset"]
        self.vocab_set = set()
        self.section_set = set()
        self.worthiness_set = set()
        self.labels_set = set()
        self.tokenizer = tfds.features.text.Tokenizer()
        self.lemmatize = True if config["lemmatize"] == "True" else False
        self.lowercase = True if config["lowercase"] == "True" else False
        self.do_balance_dataset = True if config["balance_dataset"] == "True" else False

        if self.lemmatize:
            self.nlp = spacy.load("en_core_sci_lg")

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

    def lemmatize_sentence(self, sentence):
        doc = self.nlp(sentence)

        lemmas = []

        for token in doc:
            lemmas.append(token.lemma_)

        return " ".join(lemmas)

    def preprocess_data(self, data, lemmatize=False, lowercase=False):
        if not lemmatize and not lowercase:
            return data

        for instance in data:
            sentence = instance["string"]
            if lemmatize:
                sentence = self.lemmatize_sentence(instance["string"])
            if lowercase:
                sentence.lower()

    def balance_dataset(self, dataset):
        class_background = [t for t in dataset if t["label"] == "background"]
        class_method = [t for t in dataset if t["label"] == "method"]
        class_result = [t for t in dataset if t["label"] == "result"]

        sample_size = min(len(class_background), len(class_method), len(class_result))

        bclass0 = random.sample(class_background, sample_size)
        bclass1 = random.sample(class_method, sample_size)
        bclass2 = random.sample(class_result, sample_size)

        bdataset = bclass0 + bclass1 + bclass2
        random.shuffle(bdataset)

        return bdataset

    def load_main_task_data(self, _type):
        """Loads train data for main task, no scaffolds"""
        if _type == "train":
            file = os.path.join(self.data_dir, "train.jsonl")
        elif _type == "dev":
            file = os.path.join(self.data_dir, "dev.jsonl")
        elif _type == "test":
            file = os.path.join(self.data_dir, "test.jsonl")

        with open(file, "r") as data_file:
            data = [json.loads(x) for x in list(data_file)]

        if _type in ["dev", "train"]:
            # lemmatize and lowercase data
            if self.lemmatize:
                if "lemmatized_string" in data[0]:
                    data = self.preprocess_data(data, lowercase=self.lowercase)
                else:
                    data = self.preprocess_data(
                        data, lemmatize=self.lemmatize, lowercase=self.lowercase
                    )

            # balance data
            if self.do_balance_dataset:
                data = self.balance_dataset(data)

        if self.lemmatize:
            key = "lemmatized_string"
        else:
            key = "string"

        for sample in data:
            tokens = self.tokenizer.tokenize(sample[key])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
            # self.labels_set.update([sample["label"]])
            sample["relevant_key"] = "label"
            # sample["text"] = [sample["string"]]
        return data
        # return data[:1000]  # TODO remove

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
                sample["is_citation"] = str(sample["is_citation"])
            else:
                sample["relevant_key"] = "section_title"
            tokens = self.tokenizer.tokenize(sample["text"])
            self.vocab_set.update(tokens)
            sample["tokens"] = tokens
        return data

    def load_data(self, _type, multitask=False):
        """
        load data as needed:
        :param _type:
        :param multitask: load also auxiliary tasks if True
        :return lists for text and respective class at labels, sections, worthiness
        """
        data = []

        not_encoded_tokens = []
        text = []
        labels = []
        sections = []
        worthiness = []

        if multitask:
            for i, j, k in zip(
                self.load_main_task_data(_type=_type),
                self.load_scaffold("cite"),
                self.load_scaffold("title"),
            ):
                data.append(i)
                data.append(j)
                data.append(k)
        else:
            for i in self.load_main_task_data(_type=_type):
                data.append(i)

        for sample in data:
            _text = ""
            if "text" in sample.keys():
                _text = sample["text"]
                # text.append(sample['text'])
            elif "string" in sample.keys():
                _text = sample["string"]
                # text.append(sample['string'])

            if self.lowercase:
                _text = _text.lower()

            text.append(_text)
            not_encoded_tokens.append(sample["tokens"])

            relevant_key = sample["relevant_key"]
<<<<<<< HEAD
            
=======

>>>>>>> 7a0a62b8041312c2608e5da2ca11a48b74f60341
            if relevant_key == "label":
                labels.append(sample["label"])
            else:
                labels.append("__unknown__")

            if multitask:
                if relevant_key == "section_title":
                    # avoid 'related work' to be tokenized with two labels
                    sections.append(sample["section_title"].split(" ")[0])
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
        return (
            len(self.vocab_set),
            len(self.labels_set),
            len(self.section_set),
            len(self.worthiness_set),
        )
