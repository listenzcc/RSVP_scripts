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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
import time

'''
# Function: Setting evrionment for the script.
# Output: root_path, directory of project.
# Output: time_stamp, string of beginning time of the script.
# Output: id_string, customer identifier string.
# Output: results_dir, directory for storing results.
'''
root_dir = os.path.join('D:\\', 'RSVP_MEG_experiment')
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
id_string = 'RSVP_MEG'
results_dir = os.path.join(root_dir, 'scripts', 'RSVP_MVPA', 'results')

labels = None
epochs_data = None
for i in [5, 7, 9]:
    '''
    # Function: Reading epochs from -epo.fif.
    # Output: epochs, resampled epochs.
    '''
    epo_path = os.path.join(results_dir, 'eeg_ww_epochs_%d-epo.fif' % i)

    epochs = mne.read_epochs(epo_path, verbose=True)
    epochs.crop(tmin=0.0, tmax=0.8)

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

'''
# Function: Setting MVPA stuff.
# Output: cv, cross-validation maker.
# Output: pca_pipeline, pipeline of pca decomposition.
# Output: csp_pipeline, pipeline of csp filter.
# Output: svc, SVM classifier.
'''
pca = PCA(n_components=8)

cv = ShuffleSplit(5, test_size=0.2)
pca_pipeline = make_pipeline(
    mne.decoding.Scaler(epochs.info), mne.decoding.Vectorizer(), pca)
clf = svm.SVC(gamma='scale', class_weight='balanced', C=1, tol=1e-4)
# clf = LogisticRegression(solver='liblinear', class_weight='balanced')
rfe = RFE(clf, step=1)

'''
# Function: Repeat training and testing.
# Output:
'''
for _rep in range(2):
    print(_rep)
    for _split, idxs in enumerate(cv.split(labels)):
        print(_rep, ':', _split)
        train_idx, test_idx = idxs
        # test_idx = train_idx
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_test = epochs_data[train_idx], epochs_data[test_idx]

        '''
        # Function: PCA decomposition and SVM training.
        '''
        X_train_pca = pca_pipeline.fit_transform(X_train)
        clf.fit(X_train_pca, y_train)

        '''
        # Function: PCA decomposition.
        # Output: X_test_pca, PCA decomposition testing data.
        '''
        X_test_pca = pca_pipeline.transform(X_test)

        '''
        # Function: SVM testing.
        '''
        y_predict = clf.predict(X_test_pca)
        score_acc = metrics.accuracy_score(y_test, y_predict)
        score_rec = metrics.recall_score(y_test, y_predict, average=None)
        print('PCA-SVM:', '%0.4f' % score_acc, score_rec)
