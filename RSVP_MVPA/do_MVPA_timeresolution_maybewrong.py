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

'''
# Function: Reading epochs from -epo.fif.
# Output: epochs, resampled epochs.
'''
epo_path = os.path.join(results_dir, 'eeg_ww_epochs_5-epo.fif')

epochs = mne.read_epochs(epo_path, verbose=True)
epochs.crop(tmin=0.2, tmax=0.6)

'''
# Function: Preparing dataset for MVPA.
# Output: labels, labels of each trail.
# Output: epochs_data, data of each trail.
'''
labels = epochs.events[:, -1]
epochs_data = epochs.get_data()

'''
# Function: Preparing time windows.
# Output: w_step, step of each two continus window.
# Output: w_length, length of each window.
# Output: w_start, start time of each window.
'''
sfreq = epochs.info['sfreq']
length = epochs_data.shape[2]

w_length = int(sfreq * 0.1)
w_step = int(sfreq * 0.05)
w_start = np.arange(0, length-w_length, w_step)

'''
# Function: Setting MVPA stuff.
# Output: cv, cross-validation maker.
# Output: pca_pipeline, pipeline of pca decomposition.
# Output: csp_pipeline, pipeline of csp filter.
# Output: svc, SVM classifier.
# Output: ocsvc, OneClassSVM classifier.
'''
pca = PCA(n_components=6)

cv = ShuffleSplit(5, test_size=0.2)
pca_pipeline = make_pipeline(
    mne.decoding.Scaler(epochs.info), mne.decoding.Vectorizer(), pca)
csp_pipeline = make_pipeline(
    mne.decoding.Scaler(epochs.info),
    mne.decoding.CSP(n_components=6, reg=None, log=True, norm_trace=False),
    mne.decoding.Vectorizer())
svc = svm.SVC(gamma='scale', class_weight='balanced')
ocsvm = svm.OneClassSVM(nu=0.1, gamma='scale')

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
        # Function: PCA decomposition and OneClassSVM, SVM training.
        '''
        X_train_pca = pca_pipeline.fit_transform(X_train)
        ocsvm.fit(X_train_pca[y_train == 2])
        svc.fit(X_train_pca, y_train)

        '''
        # Function: PCA decomposition.
        # Output: X_test_pca, PCA decomposition testing data.
        '''
        X_test_pca = pca_pipeline.transform(X_test)

        '''
        # Function: OneClassSVM training and testing.
        '''
        y_predict = ocsvm.predict(X_test_pca)
        y_predict = y_predict * 0.5 + 1.5
        score_acc = metrics.accuracy_score(y_test, y_predict)
        score_rec = metrics.recall_score(y_test, y_predict, average=None)
        print('PCA-OCSVM:', '%0.4f' % score_acc, score_rec)

        '''
        # Function: SVM training and testing.
        '''
        y_predict = svc.predict(X_test_pca)
        score_acc = metrics.accuracy_score(y_test, y_predict)
        score_rec = metrics.recall_score(y_test, y_predict, average=None)
        print('PCA-  SVM:', '%0.4f' % score_acc, score_rec)

        '''
        # Function: CSP decomposition and OneClassSVM, SVM training.
        '''
        X_train_csp = csp_pipeline.fit_transform(X_train, y_train)
        ocsvm.fit(X_train_csp[y_train == 2])
        svc.fit(X_train_csp, y_train)

        for i, s in enumerate(w_start):
            '''
            # Function: Croping testing data.
            # Output: X_test, cropped epochs_data.
            '''
            X_test = epochs_data[test_idx][:, :, s:(s+w_length)]

            '''
            # Function: CSP decomposition.
            # Output: X_test_csp, CSP filtered testing data.
            '''
            X_test_csp = csp_pipeline.transform(X_test)

            '''
            # Function: OneClassSVM training and testing.
            '''
            y_predict = ocsvm.predict(X_test_csp)
            y_predict = y_predict * 0.5 + 1.5
            score_acc = metrics.accuracy_score(y_test, y_predict)
            score_rec = metrics.recall_score(y_test, y_predict, average=None)
            print('%d:' % i, 'CSP-OCSVM:', '%0.4f' % score_acc, score_rec)

            '''
            # Function: SVM training and testing.
            '''
            y_predict = svc.predict(X_test_csp)
            score_acc = metrics.accuracy_score(y_test, y_predict)
            score_rec = metrics.recall_score(y_test, y_predict, average=None)
            print('%d:' % i, 'CSP-  SVM:', '%0.4f' % score_acc, score_rec)
