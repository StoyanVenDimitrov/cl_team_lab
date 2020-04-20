import jsonlines


def build_confusion_matrix(predicted, labeled):
    """
    Matrix of overlaps and mismatches
    :param predicted: list: labels the model predict for each sample
    :param labeled: list:  actual labels for sample
    :return: label_size x label_size matrix of int
    """


def get_mertrics(confusion_matrix):
    """
    get FP, FN, TP, TN: needed for precision and recall
    :param confusion_matrix
    :return: int: TP, int: TN, int: FP , int: FN
    """


def f1_score(predicted, labeled):
    """
    F1 score for prediction results
    :param predicted: list: labels the model predict for each sample
    :param labeled: list:  actual labels for sample
    :return: int: F1 score
    """
    confusion_matrix = build_confusion_matrix(predicted, labeled)
    tp, tn, fp, fn = get_mertrics(confusion_matrix)

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * ((precision * recall) / (precision + recall))
    except:
        return 0


def dev_labels(filename):
    label_list = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            label_list.append(obj['label'])
    return label_list
