import configparser
import importlib
import wandb
import os


def load_config(path):
    """
    create readable config object
    :param path: config file path
    :return: config object
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


def write_config(path, config1, config2):
    """
    join two configs
    :param path: where to write
    :param config1:
    :param config2:
    :return: joint config
    """
    with open(os.path.join(path, os.pardir, "config.txt"), "w") as f:
        f.write("[param]\n")
        for c in [config1, config2]:
            for k, v in c.items():
                f.write(k + " = " + v + "\n")


def make_filename(config):
    filename = "PARAMS=="
    for i, para in enumerate(config):
        # print(para)
        sep = "&" if i > 0 else ""
        if "model_path" in para:
            # print(para)
            continue
        elif "stopwords_path" in para:
            stopwords = config[para].split("/")[-1].split(".")[0]
            filename += f"{stopwords}-"
            # print(para)
            continue
        elif "path" in para or "dataset" == para:
            continue
        elif para == "model_version":
            config[para] = config[para].replace("/", "+")
        filename += f"{sep}{para}={config[para]}"
        # print(para)
    return filename


def make_logdir(dir, _type, config1, config2):
    filename1 = make_filename(config1)
    filename2 = make_filename(config2)

    logdir = os.path.join(
        "saved_models", dir, _type + "_" + filename1 + "_" + filename2, "logs"
    )
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    write_config(logdir, config1, config2)

    return logdir


def import_module(dotted_path):
    """
    import module from config ready to create object
    :param dotted_path: module path
    :return: imported model
    """
    module_parts = dotted_path.split(".")
    module_path = ".".join(module_parts[:-1])
    try:
        module = importlib.import_module(module_path, package=__package__)
    except ModuleNotFoundError:
        module = importlib.import_module(module_path)
    return getattr(module, module_parts[-1])


def format_time(start_seconds, end_seconds):
    seconds = end_seconds - start_seconds
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return "Days:{:0>2} Hours:{:0>2} Minutes:{:0>2} Seconds:{:05.2f}".format(
        int(days), int(hours), int(minutes), seconds
    )


def get_log_params(config):
    params = {}
    for sec in config:
        for k, v in config[sec].items():
            params[sec + "/" + k] = v
    return params


def wandb_init(config, _type="model"):
    if _type == "model":
        wandb.init(
            config={
                "model_version": config["model_version"],
                "epochs": int(config["epochs"]),
                "batch_size": int(config["batch_size"]),
                "max_len": int(config["max_len"]),
                "learning_rate": float(config["learning_rate"]),
                "bfloat16": config["bfloat16"]
            }, reinit=True
        )
    elif _type == "preprocess":
        wandb.init(
            config={
                "lemmatize": config["lemmatize"],
                "lowercase": config["lowercase"],
                "balance_dataset": config["balance_dataset"],
                "shuffle_data": config["shuffle_data"]
            }, reinit=True
        )


def wandb_init_metrics():
     wandb.init(
        config={
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "val_macro_avg_precision": 0.0,
            "val_macro_avg_recall": 0.0,
            "val_macro_avg_f1-score": 0.0,
            "val_class_0_precision": 0.0,
            "val_class_0_recall": 0.0,
            "val_class_0_f1-score": 0.0,
            "val_class_1_precision": 0.0,
            "val_class_1_recall": 0.0,
            "val_class_1_f1-score": 0.0,
            "val_class_2_precision": 0.0,
            "val_class_2_recall": 0.0,
            "val_class_2_f1-score": 0.0,
            "test_macro_avg_precision": 0.0,
            "test_macro_avg_recall": 0.0,
            "test_macro_avg_f1-score": 0.0,
            "test_class_0_precision": 0.0,
            "test_class_0_recall": 0.0,
            "test_class_0_f1-score": 0.0,
            "test_class_1_precision": 0.0,
            "test_class_1_recall": 0.0,
            "test_class_1_f1-score": 0.0,
            "test_class_2_precision": 0.0,
            "test_class_2_recall": 0.0,
            "test_class_2_f1-score": 0.0,
        }, reinit=True
    )


def wandb_log_report(_type, report):
    wandb.log(
        {
            f"{_type}_macro_avg_precision": report["macro avg"]["precision"],
            f"{_type}_macro_avg_recall": report["macro avg"]["recall"],
            f"{_type}_macro_avg_f1-score": report["macro avg"]["f1-score"],
            f"{_type}_class_0_precision": report["0"]["precision"],
            f"{_type}_class_0_recall": report["0"]["recall"],
            f"{_type}_class_0_f1-score": report["0"]["f1-score"],
            f"{_type}_class_1_precision": report["1"]["precision"],
            f"{_type}_class_1_recall": report["1"]["recall"],
            f"{_type}_class_1_f1-score": report["1"]["f1-score"],
            f"{_type}_class_2_precision": report["2"]["precision"],
            f"{_type}_class_2_recall": report["2"]["recall"],
            f"{_type}_class_2_f1-score": report["2"]["f1-score"],
        }
    )
