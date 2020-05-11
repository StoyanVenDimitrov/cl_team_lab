import configparser
import importlib

def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def make_filename(config):
    filename = ""
    for para in config:
        filename += f"{config[para]}-"
    return filename

def import_module(dotted_path):
    module_parts = dotted_path.split(".")
    module_path = ".".join(module_parts[:-1])
    try:
        module = importlib.import_module(module_path, package=__package__)
    except ModuleNotFoundError:
        module = importlib.import_module(module_path)
    return getattr(module, module_parts[-1])