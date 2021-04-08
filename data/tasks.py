import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from data.domain import Domain


def as_domain(labeled_docs, labels, unlabeled_docs, issource, domain, translations=None, language='en',
              token_pattern=r"(?u)\b\w\w+\b", min_df=1, tfidf=True):
    """
    Represents raw documents as a Domain; a domain contains the tfidf weighted co-occurrence matrices of the labeled
    and unlabeled documents (with consistent Vocabulary).
    :param labeled_docs: the set of labeled documents
    :param labels: the labels of labeled_docs
    :param unlabeled_docs: the set of unlabeled documents
    :param issource: boolean, if True then the vocabulary is bounded to the labeled documents (the training set), if
    otherwise, then the vocabulary has to be bounded to that of the unlabeled set (which is expecteldy bigger) since
    we should assume the test set is only seen during evaluation. This is not true in a Transductive setting, but we
    force it to follow the same setting so as to allow for a fair evaluation.
    :param domain: the name of the domain (e.g., 'books'
    :param language: the language of the domain (e.g., 'french')
    :param token_pattern: the token pattern the sklearn vectorizer will use to split words
    :param min_df: the minimum frequency below which words will be filtered out from the vocabulary
    :return: an instance of Domain
    """
    if issource:
        counter = CountVectorizer(token_pattern=token_pattern, min_df=min_df)
        v = counter.fit(labeled_docs).vocabulary_
        if tfidf:
            vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=token_pattern, vocabulary=v)
        else:
            vectorizer = CountVectorizer(token_pattern=token_pattern, vocabulary=v)
    else:
        if tfidf:
            vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=token_pattern, min_df=min_df)
        else:
            vectorizer = CountVectorizer(token_pattern=token_pattern, min_df=min_df)
    U = vectorizer.fit_transform(labeled_docs + unlabeled_docs)
    X = vectorizer.transform(labeled_docs)
    y = np.array(labels)
    V = vectorizer.vocabulary_
    domain = Domain(X, y, U, V, domain, language)
    if translations is not None:
        T = vectorizer.transform(translations)
        return domain, T
    else:
        return domain
