# code: utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '20190326', '20190326', 'S%02d_lixiangTHU_20190326_%02d.ds')

event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -1, 0, 4
freq_l, freq_h = 1, 270
decim = 1

# Make defaults
baseline = (tmin, t0)
reject = dict(mag=5e-12)

# Prepare rawobject
raw_files = [mne.io.read_raw_ctf(
    file_dir % (2, j), preload=True) for j in [1, 2]]
raw = mne.concatenate_raws(raw_files)
raw.filter(freq_l, freq_h, fir_design='firwin')
picks = mne.pick_types(raw.info, meg=True, exclude='bads')

# Get events
events = mne.find_events(raw, stim_channel='UPPT001')

# Get epochs
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=decim, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=baseline,
                    reject=reject, preload=True)
epochs.pick_types(meg=True, ref_meg=False)
epochs.resample(120, npad="auto")

# Get evoked
if False:
    evoked_1 = epochs['MI1'].average()
    evoked_2 = epochs['MI2'].average()
    fig, axes = plt.subplots(2, 1)
    a0 = axes[0]
    f1 = evoked_1.plot_joint(show=False)
    f2 = evoked_2.plot_joint(show=False)
    plt.show()


# MVPA
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
labels = epochs.events[:, -1]
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()


# CSP LR demo
csp = mne.decoding.CSP(n_components=6, reg=None,
                       log=True, norm_trace=False)
csp.fit_transform(epochs_data_train, labels)
csp.plot_patterns(epochs.info, show=False)
cv = ShuffleSplit(10, test_size=0.2)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(gamma='auto')
clf = make_pipeline(mne.decoding.Scaler(epochs_train.info),
                    csp,
                    mne.decoding.Vectorizer(),
                    lr)
scores = mne.decoding.cross_val_multiscore(
    clf, epochs_data_train, labels, cv=cv, n_jobs=12)
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (
    np.mean(scores), class_balance))


# MVPA time_resolution
sfreq = epochs.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
scores_rep = []
for _ in range(10):
    print(_)
    score_this_rep = []
    for train_idx, test_idx in cv.split(epochs_data):
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train = epochs_data_train[train_idx]
        clf.fit(X_train, y_train)
        score_this_window = []
        for n in w_start:
            X_test = epochs_data[test_idx][:, :, n:(n+w_length)]
            score = clf.score(X_test, y_test)
            score_this_window.append(score)
        score_this_rep.append(score_this_window)
    scores_rep.append(score_this_rep)
scores_windows = np.mean(np.array(scores_rep), 0)
# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()
