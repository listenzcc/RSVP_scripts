# coding: utf-8
'''
This script is to do MVPA on MEG RSVP dataset
'''

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import time
import pdb

'''
# Function: Setting MVPA stuff.
# Output: cv, cross-validation maker.
# Output: pca_pipeline, pipeline of pca decomposition.
# Output: xdawn_pipeline, pipeline of xdawn filter.
# Output: clf_*, classifier of svm and lr.
'''

pca = make_pipeline(mne.decoding.Vectorizer(), PCA(n_components=8))

xdawn = mne.preprocessing.Xdawn(n_components=8)

cv = StratifiedKFold(n_splits=10, shuffle=True)

normalize_pipeline = make_pipeline(mne.decoding.Vectorizer(), MinMaxScaler())

clf_svm_rbf = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced', verbose=True)


def report_results(true_label, pred_label, title=None):
        print(title)
        report = classification_report(true_label, pred_label, target_names=['odd', 'norm'])
        print(report)
        if title is None:
            return
        with open(os.path.join(results_dir, '%s.txt' % title), 'w') as f:
            f.writelines(report)


'''
# Function: Setting evrionment for the script.
# Output: root_path, directory of project.
# Output: time_stamp, string of beginning time of the script.
# Output: id_string, customer identifier string.
# Output: results_dir, directory for storing results.
'''
root_dir = os.path.join('/nfs/cell_a/userhome/zcc/documents/RSVP_experiment/')
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
id_string = 'RSVP_MEG'
results_dir = os.path.join(root_dir, 'RSVP_MVPA', 'MVPA_lr')
epochs_dir = os.path.join(root_dir, 'epochs_saver', 'epochs_freq_0.5_30_crop_n0.2_p1.1')

read_save_stuff = {}

read_save_stuff['S01'] = dict(
        range_run   = range(4, 11),
        epochs_path = os.path.join(epochs_dir, 'eeg_S01_epochs_%d-epo.fif'),
        report_path = os.path.join(results_dir, 'accs_eeg_S01.txt'))

read_save_stuff['S02'] = dict(
        range_run   = range(4, 12),
        epochs_path = os.path.join(epochs_dir, 'eeg_S02_epochs_%d-epo.fif'),
        report_path = os.path.join(results_dir, 'accs_eeg_S02.txt'))

for stuff in read_save_stuff.values():
    print('-'*80)
    for e in stuff.items():
        print(e[0], e[1])

    '''
    # Function: Reading epochs.
    '''
    labels = None
    epochs_data = None
    epochs_list = []
    for i in stuff['range_run']:
        # Function: Reading epochs from -epo.fif.
        epo_path = os.path.join(stuff['epochs_path'] % i)

        epochs = mne.read_epochs(epo_path, verbose=True)
        epochs.crop(tmin=0.0, tmax=1.0)

        # Attention!!!
        # This may cause poor alignment between epochs.
        # But this is necessary for concatenate_epochs.
        if epochs_list.__len__() != 0:
            epochs.info['dev_head_t'] = epochs_list[0].info['dev_head_t']
        epochs_list.append(epochs)

        # Function: Preparing dataset for MVPA.
        if labels is None:
            labels = epochs.events[:, -1]
            epochs_data = epochs.get_data()
        else:
            labels = np.concatenate([labels, epochs.events[:, -1]])
            epochs_data = np.concatenate([epochs_data, epochs.get_data()], 0)

    epochs = mne.epochs.concatenate_epochs(epochs_list)

    '''
    # Function: Repeat training and testing.
    # Output:
    '''
    sfreq = epochs.info['sfreq']
    w_length = int(sfreq * 0.1)   # running classifier: window length
    w_step = int(sfreq * 0.05)  # running classifier: window step size
    w_start = np.arange(0, epochs.get_data().shape[2] - w_length, w_step)


    # init preds results.
    preds_xdawn_svm_rbf = np.empty([len(labels), len(w_start)+1])
    preds_pca_svm_rbf = np.empty([len(labels), 1])

    for train, test in cv.split(epochs_data, labels):
        print('-' * 80)

        # xdawn
        xdawn_data_train = xdawn.fit_transform(epochs[train])
        xdawn_data_test = xdawn.transform(epochs[test])

        # xdawn SVM
        data_train_ = xdawn_data_train[:, :, :]
        data_test_ = xdawn_data_test[:, :, :]

        # SVM rbf
        clf_svm_rbf.fit(normalize_pipeline.fit_transform(data_train_), labels[train])
        preds_xdawn_svm_rbf[test, len(w_start)] = clf_svm_rbf.predict(
                normalize_pipeline.transform(data_test_))

        # time resolution
        for j, start in enumerate(w_start):
            print(j, start)

            data_train_ = xdawn_data_train[:, :, start:start+w_length]
            data_test_ = xdawn_data_test[:, :, start:start+w_length]

            # SVM rbf
            clf_svm_rbf.fit(normalize_pipeline.fit_transform(data_train_), labels[train])
            preds_xdawn_svm_rbf[test, j] = clf_svm_rbf.predict(
                    normalize_pipeline.transform(data_test_))

        # PCA
        pca_data_train = pca.fit_transform(epochs[train].get_data())
        pca_data_test = pca.fit_transform(epochs[test].get_data())

        # PCA SVM
        data_train_ = pca_data_train
        data_test_ = pca_data_test

        # SVM rbf
        clf_svm_rbf.fit(normalize_pipeline.fit_transform(data_train_), labels[train])
        preds_pca_svm_rbf[test, 0] = clf_svm_rbf.predict(
                normalize_pipeline.transform(data_test_))


    '''
    # Function: Save report into file.
    '''
    fpath = os.path.join(stuff['report_path'])


    with open(fpath, 'w') as f:
        report_svm_rbf = classification_report(
                preds_xdawn_svm_rbf[:, len(w_start)], labels, target_names=['odd', 'norm'])
        print(report_svm_rbf)
        f.writelines('\n[all_xdawn_SVM_rbf]\n')
        f.writelines(report_svm_rbf)

        report_svm_rbf = classification_report(
                preds_pca_svm_rbf, labels, target_names=['odd', 'norm'])
        print(report_svm_rbf)
        f.writelines('\n[all_PCA_SVM_rbf]\n')
        f.writelines(report_svm_rbf)



    for j, start in enumerate(w_start):
        print(j)

        report_svm_rbf = classification_report(
                preds_xdawn_svm_rbf[:, j], labels, target_names=['odd', 'norm'])
        with open(fpath, 'a') as f:
            print(report_svm_rbf)
            f.writelines('\n[%d-%d, %f, %f, xdawn_SVM_rbf]\n' % (
                start, start+w_length, epochs.times[start], epochs.times[start+w_length]))
            f.writelines(report_svm_rbf)

