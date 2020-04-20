import jsonlines
from collections import OrderedDict


def build_confusion_matrix(labeled, predicted):
    """
    Matrix of overlaps and mismatches
    :param predicted: list: labels the model predict for each sample
    :param labeled: list:  actual labels for sample
    :return: dict: number of overlaps between original and predicted labels, for each label
    """
    # initialize for {orig1: {pred1:number, ..}, orig2: {pred:number, ..}, ...}
    label_types = set(labeled)
    as_keys = sorted(list(label_types))
    conf_matrix = OrderedDict.fromkeys(as_keys)
    for l in conf_matrix.keys():
        conf_matrix[l] = OrderedDict.fromkeys(as_keys, 0)

    # go through the labels and increment for the matches:
    for orig, pred in zip(labeled, predicted):
        conf_matrix[orig][pred] += 1

    return conf_matrix

def get_metrics(confusion_matrix):
    """
    get FP, FN, TP, TN: needed for precision and recall
    :param confusion_matrix: as dict
    :return: dict: TP, dict: TN, dict: FP
    """
    tp, tn, fp ,fn = {}, {}, {}, {}
    #by_label = dict.fromkeys(['tp', 'fp', 'tn', 'fn'], {})

    # check label by label the types of confusion
    all_labels = confusion_matrix.keys()
    for label, row in confusion_matrix.items():
        tp[label] = row[label]
        # original label confused with predicted labels
        fn[label] = sum(row.values()) - tp[label]
        # predicted label confusing original labels:
        fp[label] = sum([confusion_matrix[l][label] for l in all_labels]) - tp[label]

    return tp, fn, fp


def f1_score(predicted, labeled):
    """
    F1 score for prediction results
    :param predicted: list: labels the model predict for each sample
    :param labeled: list:  actual labels for sample
    :return: int: F1 score
    """
    confusion_matrix = build_confusion_matrix(predicted, labeled)
    tp, fn, fp = get_metrics(confusion_matrix)

    # sum up all metric results, collapsing the labels:
    tp = sum(tp.values())
    fn = sum(fn.values())
    fp = sum(fp.values())

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


label_list = dev_labels('data/scicite/dev.jsonl')
pred_list = ['method'] * len(label_list)
f1 = f1_score(label_list, pred_list)
