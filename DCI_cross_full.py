import sys

import pandas as pd
from sklearn.metrics import classification_report

from data.domain import WordOracle
from data.tasks import as_domain
from experiments.common import pivot_selection_timed, DCIinduction
from model.dci import DCI


def set_status(progress, message=None):
    if message:
        print(message, flush=True)
    with open('status.txt', mode='wt', encoding='utf-8') as statusfile:
        print(progress, file=statusfile)


def DCI_cross_full(source_train_filename,
                   source_unlabeled_filename,
                   target_test_filename,
                   target_unlabeled_filename,
                   dictionary_filename,
                   dci_dcf,
                   dci_pivots,
                   dci_phi,
                   dci_standardization,
                   dci_optimize,
                   dci_unify,
                   dci_cross_consistency):
    set_status(2, 'Loading source training data')

    train_df = pd.read_csv(source_train_filename)
    train_df = train_df.fillna('empty text')
    source_X_str = list(train_df.iloc[:, 0])
    source_y_str = list(train_df.iloc[:, 1])

    labels = set(source_y_str)
    if len(labels) != 2:
        raise Exception('Training data must have two labels')
    label_to_id = dict()
    id_to_label = list()
    for label in labels:
        label_to_id[label] = len(label_to_id)
        id_to_label.append(label)

    source_y = [label_to_id[label] for label in source_y_str]

    if source_unlabeled_filename:
        set_status(4, 'Loading source unlabeled data')

        source_unlabeled_df = pd.read_csv(source_unlabeled_filename)
        source_unlabeled_df = source_unlabeled_df.fillna('empty text')
        source_unlabeled_str = list(source_unlabeled_df.iloc[:, 0])
    else:
        source_unlabeled_str = ['']

    set_status(6, 'Loading target test data')

    test_df = pd.read_csv(target_test_filename)
    test_df = test_df.fillna('empty text')
    target_X_str = list(test_df.iloc[:, 0])
    target_y_str = list(test_df.iloc[:, 1])

    target_y = [label_to_id[label] for label in target_y_str]

    if target_unlabeled_filename:
        set_status(8, 'Loading target unlabeled data')

        target_unlabeled_df = pd.read_csv(target_unlabeled_filename)
        target_unlabeled_df = target_unlabeled_df.fillna('empty text')
        target_unlabeled_str = list(target_unlabeled_df.iloc[:, 0])
    else:
        target_unlabeled_str = ['']

    if dictionary_filename:
        set_status(10, 'Loading dictionary')

        dictionary_df = pd.read_csv(dictionary_filename, index_col=0)
        dictionary = list(dictionary_df.to_dict().values())[0]
    else:
        dictionary = None

    set_status(12, 'Defining source domain')

    patt = r"(?u)\b\w+\b"

    source = as_domain(source_X_str, source_y, source_unlabeled_str, True, 'source', None, 'source', patt)

    set_status(21, 'Defining target domain')

    target = as_domain(target_X_str, target_y, target_unlabeled_str, False, 'target', None, 'target', patt)

    if dictionary:
        set_status(32, 'Defining WordOracle')

        oracle = WordOracle(dictionary, source, target)
    else:
        oracle = None

    set_status(33, 'Pivot selection')

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(dci_pivots, source.X, source.y, source.U, target.U, source.V,
                                                           target.V, oracle=oracle, phi=dci_phi,
                                                           cross_consistency=dci_cross_consistency)

    set_status(47, 'DCI')

    dci = DCI(dcf=dci_dcf, unify=dci_unify, post=dci_standardization)

    set_status(63, 'Doing the learn thing')

    acc, dci_time, svm_time, test_time, predictions = DCIinduction(source, target, s_pivots, t_pivots, dci,
                                                                   optimize=dci_optimize)

    set_status(95, 'Saving output')

    with open('predictions.csv', mode='wt', encoding='utf-8') as predictionsfile:
        for value in predictions:
            if value > 0.5:
                print(id_to_label[1], file=predictionsfile)
            else:
                print(id_to_label[0], file=predictionsfile)

    with open('report.txt', mode='wt', encoding='utf-8') as reportfile:
        print(f'pivot selection time = {pivot_time:.5f} secs', file=reportfile)
        print(f'dci time = {dci_time:.5f} secs', file=reportfile)
        print(f'svm time = {svm_time:.5f} secs', file=reportfile)
        print(f'test time = {test_time:.5f} secs', file=reportfile)
        print(file=reportfile)

        y_pred_clf = pd.Series([id_to_label[1] if y > 0.5 else id_to_label[0] for y in predictions],
                               name='Predicted')
        df_confusion_clf = pd.crosstab(target_y_str, y_pred_clf, margins=True)
        print(df_confusion_clf, file=reportfile)
        print(file=reportfile)
        print(classification_report(target_y, predictions > 0.5, target_names=id_to_label), file=reportfile)

    set_status(100, 'Done')


if __name__ == '__main__':

    set_status(0, 'Started')

    set_status(1, 'Input parameters' + str(list(enumerate(sys.argv))))

    if len(sys.argv) != 13:
        set_status(100,
                   'Incorrect parameters: source_train_filename source_unlabeled_filename target_test_filename '
                   'target_unlabeled_filename dictionary_filename dcf=[cosine|pmi|linear] npivots phi '
                   'standardization=[TRUE|FALSE] svm_optimization=[TRUE|FALSE] unification=[TRUE|FALSE]'
                   'cross_domain_consistency=[TRUE|FALSE]')
        exit(-1)

    source_train_filename = sys.argv[1]
    source_unlabeled_filename = sys.argv[2]
    target_test_filename = sys.argv[3]
    target_unlabeled_filename = sys.argv[4]
    dictionary_filename = sys.argv[5]
    dci_dcf = sys.argv[6]  # 'cosine','pmi','linear'
    dci_pivots = int(sys.argv[7])
    dci_phi = int(sys.argv[8])
    dci_standardization = 'normal' if sys.argv[9] == 'TRUE' else None
    dci_optimize = True if sys.argv[10] == 'TRUE' else False
    dci_unify = True if sys.argv[11] == 'TRUE' else False
    dci_cross_consistency = True if sys.argv[12] == 'TRUE' else False

    DCI_cross_full(source_train_filename,
                   source_unlabeled_filename,
                   target_test_filename,
                   target_unlabeled_filename,
                   dictionary_filename,
                   dci_dcf,
                   dci_pivots,
                   dci_phi,
                   dci_standardization,
                   dci_optimize,
                   dci_unify,
                   dci_cross_consistency)
