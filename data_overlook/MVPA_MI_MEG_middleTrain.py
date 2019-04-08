# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import time

##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
pdf_path = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                        'data_overlook', 'results',
                        'foo_%s.pdf' % time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(os.path.dirname(pdf_path)):
    os.mkdir(os.path.dirname(pdf_path))
# Parameter for read raw
file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '20190326_MI_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')
subject_name = 'zhangchuncheng'
subject_idx = 1
run_idx = [1, 2]

# Parameter for preprocess raw
freq_l, freq_h = 1, 120
fir_design = 'firwin'
meg = True
ref_meg = False
exclude = 'bads'

# Parameter for epochs
event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -1, 0, 4
freq = 120
decim = 1
reject = dict(mag=5e-12)
stim_channel = 'UPPT001'

# frequency
n_cycles = 2
num = 20

# MVPA
tmin_middle, tmax_middle = 1.0, 2.0

# multi cores
n_jobs = 12

# prepare rawobject
raw_files = [mne.io.read_raw_ctf(
    file_dir % (subject_name, subject_idx, j), preload=True) for j in run_idx]
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
epochs.resample(freq, npad="auto")

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
csp = mne.decoding.CSP(n_components=6, reg=None,
                       log=True, norm_trace=False)
# csp.fit_transform(epochs_data_train, labels)
# csp.plot_patterns(epochs.info, show=False)
cv = ShuffleSplit(10, test_size=0.2)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(gamma='auto')
lda = LinearDiscriminantAnalysis()
clf = make_pipeline(mne.decoding.Scaler(epochs_train.info),
                    csp,
                    mne.decoding.Vectorizer(),
                    lr)

# MVPA time_resolution
sfreq = epochs.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
scores_rep = []
for _ in range(10):
    print(_)
    # repeat 10 folder
    score_this_rep = []

    for train_idx, test_idx in cv.split(epochs_data):
        # for each train_set and test_set
        y_train, y_test = labels[train_idx], labels[test_idx]

        # clf training for each testing window
        score_this_window = []
        # training
        X_train = epochs_data_train[train_idx]
        clf.fit(X_train, y_train)
        for n in w_start:
            X_test = epochs_data[test_idx][:, :, n:(n+w_length)]
            # testing
            score = clf.score(X_test, y_test)
            score_this_window.append(score)

        # recording this folder
        score_this_rep.append(score_this_window)

    # recording scores in 3-D array
    scores_rep.append(score_this_rep)

# mean among the dim of 10 folder
scores_windows = np.mean(np.array(scores_rep), 0)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

figures = []
f = plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
figures.append(f)

# Saving into pdf
print('Saving into pdf.')
with PdfPages(pdf_path) as pp:
    for f in figures:
        pp.savefig(f)

# Finally, we show figures.
plt.show()
