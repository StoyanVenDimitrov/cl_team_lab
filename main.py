from src.utils import utils
from src.evaluation import custom_macro_f1_score, custom_micro_f1_score
from src.utils.reader import SciciteReader
from src.models.keras_model import MultitaskLearner
from src.models.keras_model import SingletaskLearner
import configparser
import argparse
import time

# pylint:skip-file


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def keras_multitask(self, args):
        start_time = time.time()
        
        if self.args.log_metrics:
            utils.wandb_init_logs(self.config["multitask_trainer"])

        reader = SciciteReader(self.config["preprocessor"])
        text, labels, sections, worthiness = reader.load_data(_type="train", multitask=True)

        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        keras_model = MultitaskLearner(
            self.config
        )
        text_tensor, text_tokenizer = keras_model.prepare_data(text, max_len=int(self.config["multitask_trainer"]["max_len"]))
        labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)
        sections_tensor, sections_tokenizer = keras_model.prepare_data(sections)
        worthiness_tensor, worthiness_tokenizer = keras_model.prepare_data(worthiness)

        text_tensor_dev = keras_model.prepare_dev_data(text_dev, text_tokenizer, max_len=int(self.config["multitask_trainer"]["max_len"]))
        labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)

        dataset = keras_model.create_dataset(
            text_tensor,
            labels_tensor,
            sections_tensor,
            worthiness_tensor
        )
        dev_dataset = keras_model.create_dev_dataset(
            text_tensor_dev,
            labels_tensor_dev
        )

        vocab_size = len(text_tokenizer.word_index.keys())
        labels_size = len(labels_tokenizer.word_index.keys())
        section_size = len(sections_tokenizer.word_index.keys())
        worthiness_size = len(worthiness_tokenizer.word_index.keys())

        keras_model.create_model(
            vocab_size, labels_size, section_size, worthiness_size
        )
        keras_model.fit_model(dataset, dev_dataset)

        # save model
        print("Saving model...")
        # file_name_config = utils.make_filename(config["preprocessor"])
        # file_name_model = utils.make_filename(config["multitask_trainer"])
        # # save model
        # path = os.path.join("saved_models", file_name_config+"_"+file_name_model)
        if path:
            keras_model.save_model()
            
        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))


    def keras_singletask(self, args):
        start_time = time.time()

        if self.args.log_metrics:
            utils.wandb_init_logs(self.config["singletask_trainer"])

        reader = SciciteReader(self.config["preprocessor"])
        text, labels, _, _ = reader.load_data(_type="train", multitask=False)

        text_dev, labels_dev, _, _ = reader.load_data(_type="dev", multitask=False)
        keras_model = SingletaskLearner(
            self.config
        )
        text_tensor, text_tokenizer = keras_model.prepare_data(text, max_len=int(self.config["multitask_trainer"]["max_len"]))
        labels_tensor, labels_tokenizer = keras_model.prepare_data(labels)

        text_tensor_dev = keras_model.prepare_dev_data(text_dev, text_tokenizer, max_len=int(self.config["multitask_trainer"]["max_len"]))
        labels_tensor_dev = keras_model.prepare_dev_data(labels_dev, labels_tokenizer)

        dataset = keras_model.create_dataset(
            text_tensor,
            labels_tensor
        )
        dev_dataset = keras_model.create_dev_dataset(
            text_tensor_dev,
            labels_tensor_dev
        )

        vocab_size = len(text_tokenizer.word_index.keys())
        labels_size = len(labels_tokenizer.word_index.keys())

        keras_model.create_model(
            vocab_size, labels_size
        )
        keras_model.fit_model(dataset, dev_dataset)

        # save model
        print("Saving model...")
        # file_name_config = utils.make_filename(config["preprocessor"])
        # file_name_model = utils.make_filename(config["singletask_trainer"])
        # # save model
        # path = os.path.join("saved_models", file_name_config+"_"+file_name_model)
        if path:
            keras_model.save_model()

        end_time = time.time()
        total_time = end_time - start_time
        print("Execution time:", str(datetime.timedelta(seconds=total_time)))


def run(args, config):
    if args.train:
        trainer = Trainer(args, config)
        if "multitask_trainer" in config:
            trainer.keras_multitask(args)
        if "singletask_trainer" in config:
            trainer.keras_singletask(args)

    # # evaluate on test set
    # text_test, labels_test, _, _ = reader.load_data(_type="test", multitask=False)
    #
    # text_tensor_dev = keras_model.prepare_dev_data(text_test, text_tokenizer)
    # labels_tensor_dev = keras_model.prepare_dev_data(labels_test, labels_tokenizer)
    #
    # keras_model.evaluate(text_tensor_dev, labels_tensor_dev)



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
        "--log_metrics",
        required=False,
        default=False,
        help="Set to True if metrics should me logged with mlflow, else set to False.",
    )
    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("configs/default.conf")
    
    run(args, config)  # main function

    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time:", str(datetime.timedelta(seconds=total_time)))

    # trainer = Trainer(args)
    # if args.train_features:
    #     trainer.train_feature_extractor()
    # if args.train:
    #     trainer.train_classifier()
    # macro_f1, micro_f1 = trainer.evaluate()
    # print(f"Macro F1: {macro_f1}\nMicro F1: {micro_f1}")
    #
    # # if args.test:
    # # predictor = Predictor()
    # # print(predictor.predict("Set to True if testing, else set to False."))
