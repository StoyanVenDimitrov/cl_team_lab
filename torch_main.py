import nlp
import torch

import random
import argparse
import configparser

from src.models.albert_torch import AlbertModel
from src.utils.utils import wandb_init


def balance_data(dataset):
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
    random.shuffle(bidxs)

    def keep(instance, idx):
        return idx in bidxs

    bdataset = dataset.filter(keep, with_indices=True, load_from_cache_file=False)

    return bdataset


def main(args):
    config = configparser.ConfigParser()
    config.read("configs/default.conf")

    if args.log_metrics:
        wandb_init(config["albert_torch"])

    scicite = nlp.load_dataset("scicite")
    train, dev, test = scicite["train"], scicite["validation"], scicite["test"]

    if config["albert_torch"]["balance_data"] == "True":
        train, dev = balance_data(train), balance_data(dev)

    if config["albert_torch"]["shuffle_data"] == "True":
        pass

    if config["albert_torch"]["lemmatize"] == "True":
        pass

    model = AlbertModel(config["albert_torch"])
    train_dataloader, dev_dataloader, test_dataloader = model.prepare_data(train, dev, test, int(config["albert_torch"]["batch_size"]))

    if args.train:
        model.train(train_dataloader, dev_dataloader)

    if args.test:
        model.evaluate(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        required=True,
        default=False,
        help="Set to True if training, else set to False.",
    )
    parser.add_argument(
        "--test",
        required=False,
        default=False,
        help="Set to True if testing, else set to False.",
    )
    parser.add_argument(
        "--log-metrics",
        required=False,
        default=True,
        help="Set to True if metrics should be logged, else set to False.",
    )
    args = parser.parse_args()

    main(args)

