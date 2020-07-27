import time
import datetime
import random
import argparse
import configparser

from src.models.torch_model import TransformerModel
from src.utils.reader import SciciteReaderNLP
from src.utils import utils


def run(args, config):

    reader = SciciteReaderNLP(config["preprocessor"])
    train, dev, test = reader.preprocess_data()

    model = TransformerModel(args, config)
    train_dataloader, dev_dataloader, test_dataloader = model.prepare_data(train, dev, test, int(config["torch"]["batch_size"]))

    if args.train:
        model.train(train_dataloader, dev_dataloader)

    if args.test:
        model.evaluate(test_dataloader)


if __name__ == "__main__":
    start_time = time.time()
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
        default=True,
        help="Set to True if testing, else set to False.",
    )
    parser.add_argument(
        "--log-metrics",
        required=False,
        default=True,
        help="Set to True if metrics should be logged, else set to False.",
    )
    parser.add_argument(
        "--config",
        required=False,
        default="configs/torch/default.conf",
        help="Path to configuration file.",
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print("Loading configuration from", args.config)

    run(args, config)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time:", str(datetime.timedelta(seconds=total_time)))
