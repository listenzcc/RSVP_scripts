# code: utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import time

'''
This script is to score EEG Motiong Imaging data.
Scoring is using MVPA and cross-validation.
Training data and testing data is matched in time crop,
so names as eachTrain.
There are several filter parameters in preprocessing.
Using csp for feature extraction.
Using LR, SVM and LDA as classifier.
Saving scores into npz files.
'''

##############
# Parameters #
##############
time_stamp = time.strftime('MI_EEG_eachTrain_%Y-%m-%d-%H-%M-%S')
print('Initing parameters.')
# Results pdf path
result_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                          'MotionImaging', 'results', time_stamp)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

npz_path = os.path.join(result_dir, 'npz_%s.npz')

# Parameter for read raw
file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '%s', '%s')
subject_name = '20190402_MI_EEG_maxuelin'
cnt_files = ['mxl_MI_1.cnt', 'mxl_MI_2.cnt']

freq_l = 7
for freq_h in [30, 60, 120]:
    # Parameter for preprocess raw
    fir_design = 'firwin'
    meg = False
    ref_meg = False
    eeg = True
    exclude = 'bads'

    # Parameter for epochs
    event_id = dict(MI1=1, MI2=2)
    tmin, t0, tmax = -1, 0, 4
    freq_resample = 240
    decim = 1
    reject = dict()
    stim_channel = 'STI 014'

    # frequency
    n_cycles = 2
    num = 20

    # MVPA
    tmin_middle, tmax_middle = 1.0, 2.0
    n_components = 6
    repeat_times = 10
    n_folder = 10

    # multi cores
    n_jobs = 12

    # prepare rawobject
    motage = mne.channels.read_montage('standard_1020')
    raw_files = [mne.io.read_raw_cnt(
        file_dir % (subject_name, j), motage, preload=True) for j in cnt_files]
    raw = mne.concatenate_raws(raw_files)
    raw.filter(freq_l, freq_h, fir_design=fir_design)
    # choose channel type
    picks = mne.pick_types(raw.info, eeg=eeg, exclude=exclude)

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
    epochs.resample(freq_resample, npad="auto")

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
    svm = SVC(gamma='auto')
    lda = LinearDiscriminantAnalysis()

    # MVPA time_resolution
    # make windows
    sfreq = epochs.info['sfreq']
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
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

    for rep in range(repeat_times):
        # for each repeat
        print(rep)
        # n_folder cross validation
        for split, idxs in enumerate(cv.split(epochs_data)):
            print(rep, ':', split)
            train_idx, test_idx = idxs
            # labels
            y_train, y_test = labels[train_idx], labels[test_idx]
            for clf_name in clf_dict.keys():
                print(clf_name)
                clf_pipeline = clf_pipelines[clf_name]
                for w, n in enumerate(w_start):
                    # training data
                    X_train = epochs_data[train_idx][:, :, n:(n+2*w_length)]
                    # fit
                    clf_pipeline.fit(X_train, y_train)
                    # testing data
                    X_test = epochs_data[test_idx][:, :, n:(n+w_length)]
                    # score
                    scores[clf_name][rep, split, w] = clf_pipeline.score(
                        X_test, y_test)

    # time line
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    # save into npz file
    np.savez(npz_path % 'l_%0.1f_h_%0.2f' % (freq_l, freq_h), 
        cores=scores, w_times=w_times)
