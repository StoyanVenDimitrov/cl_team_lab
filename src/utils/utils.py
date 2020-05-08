import configparser


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def make_filename(config):
    filename = ""
    for para in config:
        filename += f"{config[para]}-"
    return filename
