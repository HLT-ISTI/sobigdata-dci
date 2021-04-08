import pickle
import tarfile

from bs4 import BeautifulSoup as bs

from util.file import *


def _proc_review(doc):
    parts = doc.split(' ')
    label = parts[-1].replace('#label#:', '').strip()
    assert label in ['positive', 'negative'], 'error parsing label {}'.format(label)
    label = 1 if label == 'positive' else 0
    repeat_word = lambda word, num: ' '.join([word] * int(num))
    text = ' '.join([repeat_word(*x.split(':')) for x in parts[:-1]])
    return text, label


def fetch_Webis_cls_10(DOWNLOAD_URL='https://zenodo.org/record/3251672/files/cls-acl10-processed.tar.gz?download=1',
                       # DOWNLOAD_URL = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-cls-10/cls-acl10-processed.tar.gz',
                       dataset_home='../datasets/Webis-CLS-10',
                       max_documents=50000,
                       languages=['de', 'en', 'fr', 'jp'],
                       domains=None,
                       skip_translations=False,
                       dopickle=False):
    """
    Fetchs the processed version of the Webis-CLS-10 dataset for cross-lingual adaptation defined in:
    Prettenhofer, Peter, and Benno Stein.
    "Cross-language text classification using structural correspondence learning."
    Proceedings of the 48th annual meeting of the association for computational linguistics.
    Association for Computational Linguistics, 2010.
    """
    picklepath = join(dataset_home, 'webiscls10_processed.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    dataset_path = join(dataset_home, 'cls-acl10-processed.tar.gz')
    create_if_not_exist(dataset_home)

    if not exists(dataset_path):
        print("downloading Webis-CLS1-0 dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    dataset_path = dataset_path.replace('.tar.gz', '')

    documents = dict()
    for language in languages:
        documents[language] = dict()
        domain_list = domains if domains is not None else list_dirs(join(dataset_path, language))
        for domain in domain_list:
            documents[language][domain] = dict()
            for file in ['train.processed', 'test.processed', 'unlabeled.processed']:
                documents[language][domain][file] = []
                for doc in open(join(dataset_path, language, domain, file), 'rt', encoding='utf-8'):
                    text, label = _proc_review(doc)
                    documents[language][domain][file].append((text, label))
                    if max_documents is not None and len(documents[language][domain][file]) >= max_documents:
                        break
                print('{} documents read for language {}, domain {}, in file {}'.format(
                    len(documents[language][domain][file]), language, domain, file))

    translations = dict()
    if not skip_translations:
        for language in ['de', 'fr', 'jp']:
            translations[language] = dict()
            for domain in list_dirs(join(dataset_path, language)):
                translations[language][domain] = dict()
                for file in ['test.processed']:
                    translations[language][domain][file] = []
                    for doc in open(join(dataset_path, language, domain, 'trans', 'en', domain, file), 'rt',
                                    encoding='utf-8'):
                        text, label = _proc_review(doc)
                        translations[language][domain][file].append((text, label))
                    print('{} translations read for language {}, domain {}, in file {}'.format(
                        len(translations[language][domain][file]), language, domain, file))

    dictionaries = dict()
    split_t = lambda x: x.strip().replace(' ', '').split('\t')
    for d in list_files(join(dataset_path, 'dict')):
        dictionaries[d] = dict(map(split_t, open(join(dataset_path, 'dict', d), encoding='utf-8')))

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump((documents, translations, dictionaries), open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return documents, translations, dictionaries


def fetch_Webis_cls_10_unprocessed(
        DOWNLOAD_URL='https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz?download=1',
        # DOWNLOAD_URL = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-cls-10/cls-acl10-unprocessed.tar.gz',
        dataset_home='../datasets/Webis-CLS-10',
        max_documents=50000,
        languages=['de', 'en', 'fr', 'jp'],
        domains=None,
        dopickle=False):
    """
    Fetchs the unprocessed version of the Webis-CLS-10 dataset for cross-lingual adaptation defined in:
    Prettenhofer, Peter, and Benno Stein.
    "Cross-language text classification using structural correspondence learning."
    Proceedings of the 48th annual meeting of the association for computational linguistics.
    Association for Computational Linguistics, 2010.
    """
    picklepath = join(dataset_home, 'webiscls10_unprocessed.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    dataset_path = join(dataset_home, 'cls-acl10-unprocessed.tar.gz')
    create_if_not_exist(dataset_home)

    if not exists(dataset_path):
        print("downloading Webis-CLS1-0 dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    dataset_path = dataset_path.replace('.tar.gz', '')

    documents = dict()
    for language in languages:
        documents[language] = dict()
        domain_list = domains if domains is not None else list_dirs(join(dataset_path, language))
        for domain in domain_list:
            documents[language][domain] = dict()
            for file in ['train.review', 'test.review', 'unlabeled.review']:
                documents[language][domain][file] = []
                with open(join(dataset_path, language, domain, file), 'rt', encoding='utf-8') as doc:
                    content = bs(doc.read(), features="xml")
                for item in content.find_all('item'):
                    rating = float(item.find('rating').text)
                    if rating >= 4.0:
                        label = 'positive'
                    elif rating < 3.0:
                        label = 'negative'
                    else:
                        continue
                    text = item.find('text').text
                    documents[language][domain][file].append((text, label))
                if max_documents is not None and len(documents[language][domain][file]) >= max_documents:
                    break
                print('{} documents read for language {}, domain {}, in file {}'.format(
                    len(documents[language][domain][file]), language, domain, file))

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump(documents, open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return documents
