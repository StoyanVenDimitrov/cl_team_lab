import configparser
import importlib


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def make_filename(config):
    filename = ""
    for i, para in enumerate(config):
        sep = "&" if i > 0 else ""
        if "model_path" in para:
            continue
        elif "stopwords_path" in para:
            stopwords = config[para].split("/")[-1].split(".")[0]
            filename += f"{stopwords}-"
            continue
        filename += f"{sep}{para}={config[para]}"
    return filename


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
            res[sec + "/" + k] = v
    return params
