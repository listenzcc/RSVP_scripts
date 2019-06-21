# coding: utf-8
'''
This script is to do MVPA on MEG RSVP dataset
'''

import mne
import numpy as np
import os
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
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
root_dir = os.path.join('d:\\', 'RSVP_MEG_experiment')
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
time_stamp = '0.5-30'
id_string = 'RSVP_MEG'
results_dir = os.path.join(root_dir, 'RSVP_MVPA')
epochs_dir = os.path.join(root_dir, 'epochs_saver', 'epochs_0.5-30')

labels = None
epochs_data = None
epochs_list = []
for i in [5, 7, 9]:  # range(4, 12):  # range(1, 11):  # [5, 7, 9]:
    '''
    # Function: Reading epochs from -epo.fif.
    # Output: epochs, resampled epochs.
    '''
    epo_path = os.path.join(epochs_dir, 'eeg_mxl_epochs_%d-epo.fif' % i)

    epochs = mne.read_epochs(epo_path, verbose=True)
    epochs.crop(tmin=0.2, tmax=0.5)

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

epochs = mne.epochs.concatenate_epochs(epochs_list)


'''
# Function: Setting MVPA stuff.
# Output: cv, cross-validation maker.
# Output: pca_pipeline, pipeline of pca decomposition.
# Output: xdawn_pipeline, pipeline of xdawn filter.
# Output: svc, SVM classifier.
'''

xdawn = mne.preprocessing.Xdawn(n_components=8)

cv = StratifiedKFold(n_splits=10, shuffle=True)

normalize_pipeline = make_pipeline(mne.decoding.Vectorizer(),
                                   MinMaxScaler())

clf = svm.SVC(gamma='scale', class_weight='balanced', verbose=True)

'''
# Function: Repeat training and testing.
# Output:
'''

sfreq = epochs.info['sfreq']
w_length = int(sfreq * 0.1)   # running classifier: window length
w_step = int(sfreq * 0.05)  # running classifier: window step size
w_start = np.arange(0, epochs.get_data().shape[2] - w_length, w_step)


def report_results(true_label, pred_label, title=None, time_stamp=time_stamp):
    print(title)
    report = classification_report(
        true_label, pred_label, target_names=['odd', 'norm'])
    print(report)
    if title is None:
        return
    with open(os.path.join(results_dir,
                           '%s_%s.txt' % (title, time_stamp)), 'w') as f:
        f.writelines(report)


preds_xdawn = np.empty([len(labels), len(w_start)])

for train, test in cv.split(epochs_data, labels):
    print('-' * 80)

    # xdawn -SVM
    xdawn_data_train = xdawn.fit_transform(epochs[train])
    xdawn_data_test = xdawn.transform(epochs[test])

    for j, start in enumerate(w_start):
        print(j, start)

        data_train_ = xdawn_data_train[:, :, start:start+w_length]
        data_test_ = xdawn_data_test[:, :, start:start+w_length]

        clf.fit(normalize_pipeline.fit_transform(data_train_), labels[train])
        preds_xdawn[test, j] = clf.predict(
            normalize_pipeline.transform(data_test_))


for j, start in enumerate(w_start):
    print(j)
    report_results(labels, preds_xdawn[:, j])
