import sys

from DCI_cross_full import set_status, DCI_cross_full

if __name__ == '__main__':

    set_status(0, 'Started')

    set_status(1, 'Input parameters' + str(list(enumerate(sys.argv))))

    if len(sys.argv) != 12:
        set_status(100,
                   'Incorrect parameters: source_train_filename source_unlabeled_filename target_test_filename '
                   'target_unlabeled_filename dcf=[cosine|pmi|linear] npivots phi '
                   'standardization=[TRUE|FALSE] svm_optimization=[TRUE|FALSE] unification=[TRUE|FALSE]'
                   'cross_domain_consistency=[TRUE|FALSE]')
        exit(-1)

    source_train_filename = sys.argv[1]
    source_unlabeled_filename = sys.argv[2]
    target_test_filename = sys.argv[3]
    target_unlabeled_filename = sys.argv[4]
    dci_dcf = sys.argv[5]  # 'cosine','pmi','linear'
    dci_pivots = int(sys.argv[6])
    dci_phi = int(sys.argv[7])
    dci_standardization = 'normal' if sys.argv[8] == 'TRUE' else None
    dci_optimize = True if sys.argv[9] == 'TRUE' else False
    dci_unify = True if sys.argv[10] == 'TRUE' else False
    dci_cross_consistency = True if sys.argv[11] == 'TRUE' else False

    DCI_cross_full(source_train_filename,
                   source_unlabeled_filename,
                   target_test_filename,
                   target_unlabeled_filename,
                   None,
                   dci_dcf,
                   dci_pivots,
                   dci_phi,
                   dci_standardization,
                   dci_optimize,
                   dci_unify,
                   dci_cross_consistency)
