from collections import defaultdict


def bow(text, threshold=5):
    """
    Function to generate bag-of-words representation of text.
    :param text: list of strings (Each string should be a sentence.)
    :param threshold: int, minimum number of occurrences for words
    :return: list: list of lists containing integers
    """

    tokens = " ".join(text).lower().split(" ")

    d = defaultdict(int)
    for w in tokens:
        d[w] += 1

    d_sorted = sorted(d.items(), key=lambda k: k[1], reverse=True)

    words = [w for w, _ in d_sorted[:5]]

    bow = []
    for sent in text:
        row = [0] * len(words)
        for i, w in enumerate(sent.lower().split(" ")):
            idx = words.index(w) if w in words else None
            if idx is not None:
                row[idx] += 1
        bow.append(row)

    return bow