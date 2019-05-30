# coding: utf-8
'''
This script is to do MVPA on MEG RSVP dataset
'''

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import time

'''
# Function: Setting evrionment for the script.
# Output: root_path, directory of project.
# Output: time_stamp, string of beginning time of the script.
# Output: id_string, customer identifier string.
# Output: results_dir, directory for storing results.
'''
root_dir = os.path.join('/', 'nfs', 'cell_a', 'userhome', 'zcc', 'Documents', 'RSVP_MEG_experiment')

time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
id_string = 'RSVP_MEG'
results_dir = os.path.join(root_dir, 'scripts', 'RSVP_MVPA', 'results')

labels = None
epochs_data = None
epochs_list = []
for i in range(1, 11):
    '''
    # Function: Reading epochs from -epo.fif.
    # Output: epochs, resampled epochs.
    '''
    epo_path = os.path.join(results_dir, 'eeg_mxl_epochs_%d-epo.fif' % i)

    epochs = mne.read_epochs(epo_path, verbose=True)
    epochs.crop(tmin=0.0, tmax=0.8)

    # Attention!!!
    # This may cause poor alignment between epochs.
    # But this is necessary for concatenate_epochs.
    if epochs_list.__len__() != 0:
        epochs.info['dev_head_t'] = epochs_list[0].info['dev_head_t']
    epochs_list.append(epochs)

    '''
    # Function: Preparing dataset for MVPA.
    # Output: labels, labels of each trail.
    # Output: epochs_data, data of each trail.
    '''
    if labels is None:
        labels = epochs.events[:, -1]
        epochs_data = epochs.get_data()
    else:
        labels = np.concatenate([labels, epochs.events[:, -1]])
        epochs_data = np.concatenate([epochs_data, epochs.get_data()], 0)
epochs = mne.concatenate_epochs(epochs_list)

'''
# Function: Setting MVPA stuff.
# Output: cv, cross-validation maker.
# Output: pca_pipeline, pipeline of pca decomposition.
# Output: csp_pipeline, pipeline of csp filter.
# Output: svc, SVM classifier.
'''
pca = PCA(n_components=8)
csp = mne.decoding.CSP(n_components=8)
xdawn = mne.preprocessing.Xdawn(n_components=6)

cv = ShuffleSplit(10, test_size=0.1)
pca_pipeline = make_pipeline(mne.decoding.Scaler(epochs.info),
                             mne.decoding.Vectorizer(),
                             pca,
                             MinMaxScaler())
csp_pipeline = make_pipeline(mne.decoding.Scaler(epochs.info),
                             csp,
                             mne.decoding.Vectorizer())
xdawn_pipeline = make_pipeline(xdawn,
                               mne.decoding.Vectorizer(),
                               MinMaxScaler())
clf = svm.SVC(gamma='scale', class_weight='balanced', C=1, tol=1e-4)
rfe = RFE(clf, step=1)

'''
# Function: Repeat training and testing.
# Output:
'''
num_rep = 13
pca_score_acc = np.zeros([num_rep, cv.n_splits])
pca_score_rec = np.zeros([num_rep, cv.n_splits, 2])
xdawn_score_acc = np.zeros([num_rep, cv.n_splits])
xdawn_score_rec = np.zeros([num_rep, cv.n_splits, 2])
for _rep in range(num_rep):
    print(_rep)
    for _split, idxs in enumerate(cv.split(labels)):
        print(_rep, ':', _split)
        train_idx, test_idx = idxs
        # test_idx = train_idx
        e_train, e_test = epochs[train_idx], epochs[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_test = epochs_data[train_idx], epochs_data[test_idx]

        '''
        # Function: PCA decomposition and SVM discriminative analysis.
        '''
        X_train_pca = pca_pipeline.fit_transform(X_train)
        clf.fit(X_train_pca, y_train)
        X_test_pca = pca_pipeline.transform(X_test)
        y_predict = clf.predict(X_test_pca)
        score_acc = metrics.accuracy_score(y_test, y_predict)
        score_rec = metrics.recall_score(y_test, y_predict, average=None)
        print('PCA-SVM:', '%0.4f' % score_acc, score_rec)
        pca_score_acc[_rep][_split] = score_acc
        pca_score_rec[_rep][_split] = score_rec

        '''
        # Function: Xdawn decomposition and SVM discriminative analysis.
        '''
        X_train_xdawn = xdawn_pipeline.fit_transform(e_train, y_train)
        clf.fit(X_train_xdawn, y_train)
        X_test_xdawn = xdawn_pipeline.transform(e_test)
        y_predict = clf.predict(X_test_xdawn)
        score_acc = metrics.accuracy_score(y_test, y_predict)
        score_rec = metrics.recall_score(y_test, y_predict, average=None)
        print('Xdawn-SVM:', '%0.4f' % score_acc, score_rec)
        xdawn_score_acc[_rep][_split] = score_acc
        xdawn_score_rec[_rep][_split] = score_rec


def print_acc(acc):
    print('-' * 60)
    print(np.mean(acc))


def print_rec(rec):
    print('-' * 60)
    print(np.mean(np.mean(rec, 0), 0))


print_acc(pca_score_acc)
print_rec(pca_score_rec)
print_acc(xdawn_score_acc)
print_rec(xdawn_score_rec)

np.save('pca_score_acc.npy', pca_score_acc)
np.save('pca_score_rec.npy', pca_score_rec)
np.save('xdawn_score_acc.npy', xdawn_score_acc)
np.save('xdawn_score_rec.npy', xdawn_score_rec)

