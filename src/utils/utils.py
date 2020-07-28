import configparser
import importlib
import os


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def write_config(path, config1, config2):
    with open(os.path.join(path, os.pardir, "config.txt"), "w") as f:
        f.write("[param]\n")
        for c in [config1, config2]:
            for k, v in c.items():
                f.write(k + " = " + v + "\n")


def make_filename(config):
    filename = "PARAMS=="
    for i, para in enumerate(config):
        sep = "&" if i > 0 else ""
        if "model_path" in para:
            continue
        elif "stopwords_path" in para:
            stopwords = config[para].split("/")[-1].split(".")[0]
            filename += f"{stopwords}-"
            continue
        elif "path" in para or "dataset" in para:
            continue
        elif para == "model_version":
            if "/" in para:
                para = para.replace("/", "_")
        filename += f"{sep}{para}={config[para]}"
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
