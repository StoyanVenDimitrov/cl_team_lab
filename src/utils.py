from collections import defaultdict


def split_into_sentences(text):
    """
    Splits a text into sentences.
    :param text:
    :return:
    """
    return text.split(". ")


def remove_stopwords(text, stopwords_path):
    """
    Removes stopwords from a text.
    :param text:
    :param stopwords_path:
    :return:
    """
    if stopwords_path:
        with open(stopwords_path, 'r') as f:
            stopwords = f.read().split('\n')
        text = " ".join([w for w in text.split(" ") if w not in stopwords])
    return text


def get_word_frequencies(tokens, sort=False, descending=True):
    """
    Counts word frequencies in a text.
    :param tokens:
    :param sort:
    :param descending:
    :return:
    """
    d = defaultdict(int)

    for t in tokens:
        d[t] += 1

    if sort:
        d = sorted(d.items(), key=lambda k: k[1], reverse=descending)

    return d


def bow(text, threshold=None, min_occurrence=None, sort=True, stopwords_path=None):
    """
    Function to generate bag-of-words representation of text.
    :param text: string, this should be the entire text in a document
    :param threshold: int, number to choose the top words sorted word frequency list
    :param min_occurrence: int, minimum number of occurrences for words
    :param sort: bool, parameter for sorting word frequency
    :param stopwords_path: str, path to stopwords file
    :return: list: list of lists containing integers
    """

    text = text.lower()

    text = remove_stopwords(text, stopwords_path)

    sentences = split_into_sentences(text)

    tokens = " ".join(sentences).split(" ")

    word_frequencies = get_word_frequencies(tokens, sort=sort)

    if min_occurrence:
        word_frequencies = [x for x in word_frequencies if x[1] > min_occurrence]

    # pick the top n
    # TODO pick words that occur more than n
    if threshold:
        words = [w for w, _ in word_frequencies[:threshold]]
    else:
        words = [w for w, _ in word_frequencies]

    # generate bow
    bow = []
    for sent in sentences:
        row = [0] * len(words)
        for i, w in enumerate(sent.lower().split(" ")):
            idx = words.index(w) if w in words else None
            if idx is not None:
                row[idx] += 1
        bow.append(row)

    for i, b in enumerate(bow):
        print(sentences[i])
        print(b, '\n')

    return bow