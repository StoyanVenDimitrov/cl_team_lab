import abc


class FeatureExtractorModule:
    """Abstract class for feature extractors."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def vectorize(self, text):
        """
        Convert text into a vector representation.
        :param text:
        :return:
        """
        return

    @abc.abstractmethod
    def save_model(self):
        """
        save the model to reuse it for inference
        :return:
        """
        return

    @abc.abstractmethod
    def load_model(self, path):
        """
        load saved model
        :return: model
        """
        return
