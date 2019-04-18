# code: utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import time
import sys

from copy import deepcopy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import metrics

'''
This script is to score MEG RSVP data.
Scoring is using MVPA and cross-validation.
Training data is cropped from 1.0~2.0 seconds, so names as middleTrain.
There are several filter parameters in preprocessing.
Using csp for feature extraction.
Using LR, SVM and LDA as classifier.
Saving scores into npz files.
'''

##############
# Parameters #
##############
time_stamp = time.strftime('RSVP_MEG_middleTrain_%Y-%m-%d-%H-%M-%S')
print('Initing parameters.')
# Results pdf path
root_path = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts', 'RSVP')
result_dir = os.path.join(root_path, 'results', time_stamp)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

npz_path = os.path.join(result_dir, 'npz_%s.npz')

# Parameter for read raw
file_dir = os.path.join(root_path, '..', '..', 'rawdata',
                        '20190326_RSVP_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')
subject_name = 'maxuelin'
subject_idx = 2
# if len(sys.argv) > 1:
#     subject_name = sys.argv[1]
#     subject_idx = int(sys.argv[2])
# run_idx = [e for e in range(4, 11)]
run_idx = [5, 7]

# Parameter for preprocess raw
freq_l, freq_h = 0.1, 50
fir_design = 'firwin'
meg = True
ref_meg = False
exclude = 'bads'

# Parameter for epochs
event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -0.2, 0, 1
freq = 180
decim = 1
reject = dict(mag=5e-12)
stim_channel = 'UPPT001'

# frequency
n_cycles = 2
num = 20

# MVPA
tmin_middle, tmax_middle = 0.1, 0.3
n_components = 6
repeat_times = 10
n_folder = 10

# multi cores
n_jobs = 12

# prepare rawobject
raw_files = [mne.io.read_raw_ctf(
    file_dir % (subject_name, subject_idx, j), preload=True)
    for j in run_idx]
raw = mne.concatenate_raws(raw_files)
raw.filter(freq_l, freq_h, fir_design=fir_design)
# choose channel type
picks = mne.pick_types(raw.info, meg=meg, ref_meg=ref_meg, exclude=exclude)

#############
# Let it go #
#############
# Get epochs
print('Getting epochs.')
events = mne.find_events(raw, stim_channel=stim_channel)
baseline = (tmin, t0)
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=decim, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=baseline,
                    reject=reject, preload=True)
epochs.resample(125, npad="auto")

# Exclude abscent channels in layout
ch_names = [e[0:5] for e in epochs.ch_names]
layout_all = mne.find_layout(epochs.info)
ex = [e for e in layout_all.names if e not in ch_names]
print('Exclude channels are', ex)
# layout_all.plot(picks=[layout_all.names.index(e) for e in ex])
layout = mne.find_layout(epochs.info, exclude=ex)

# MVPA
epochs_train = epochs.copy().crop(tmin=tmin_middle, tmax=tmax_middle)
labels = epochs.events[:, -1]
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

# CSP LR demo
csp = mne.decoding.CSP(n_components=n_components, reg=None,
                       log=True, norm_trace=False)
# csp.fit_transform(epochs_data_train, labels)
# csp.plot_patterns(epochs.info, show=False)
cv = ShuffleSplit(n_folder, test_size=0.2)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(gamma='auto', probability=True)
lda = LinearDiscriminantAnalysis()

# MVPA time_resolution
# make windows
sfreq = epochs.info['sfreq']
w_length = int(sfreq * 0.1)   # running classifier: window length
w_step = int(sfreq * 0.05)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

print('repeat_times:', repeat_times)
print('n_folder:', cv.get_n_splits())
print('windows_number:', len(w_start))

# prepare clf_pipelines and scores
clf_dict = dict(lr=lr, svm=svm, lda=lda)
clf_pipelines = dict()
scores = dict()
for clf in clf_dict.items():
    clf_pipelines[clf[0]] = make_pipeline(
        mne.decoding.Scaler(epochs_train.info),
        csp, mne.decoding.Vectorizer(), clf[1])
    scores[clf[0]] = np.zeros(
        [repeat_times, cv.get_n_splits(), len(w_start)])
# sccuracy
scores_Acc = deepcopy(scores)
# recall
scores_Rec = deepcopy(scores)
# area under roc
scores_Auc = deepcopy(scores)

for rep in range(repeat_times):
    # for each repeat
    print(rep)
    # n_folder cross validation
    for split, idxs in enumerate(cv.split(epochs_data)):
        print(rep, ':', split)
        train_idx, test_idx = idxs
        # labels
        y_train, y_test = labels[train_idx], labels[test_idx]
        # training data
        X_train = epochs_data_train[train_idx]
        for clf_name in clf_dict.keys():
            print(clf_name)
            clf_pipeline = clf_pipelines[clf_name]
            # fit
            clf_pipeline.fit(X_train, y_train)
            for w, n in enumerate(w_start):
                # testing data
                X_test = epochs_data[test_idx][:, :, n:(n+w_length)]

                # predicting data
                y_prob = clf_pipeline.predict_proba(X_test)
                y_predict = y_test * 0 + 2
                for j in range(len(y_predict)):
                    if y_prob[j][0] > y_prob[j][1]:
                        y_predict[j] = 1

                # scoring acc
                score_acc = metrics.accuracy_score(y_test, y_predict)
                # scoring rec
                score_rec = metrics.recall_score(y_test, y_predict)
                # scoring auc
                score_auc = metrics.roc_auc_score(y_test == 1, y_prob[:, 0])

                # storing scores
                scores_Acc[clf_name][rep, split, w] = score_acc
                scores_Rec[clf_name][rep, split, w] = score_rec
                scores_Auc[clf_name][rep, split, w] = score_auc


# time line
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

# save into npz file
np.savez(npz_path % str(freq),
         scores_Acc=scores_Acc,
         scores_Rec=scores_Rec,
         scores_Auc=scores_Auc,
         w_times=w_times)
