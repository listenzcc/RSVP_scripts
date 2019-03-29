# code: utf-8

import matplotlib.pyplot as plt
import mne
import os
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score


file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '20190326', '20190326', 'S%02d_lixiangTHU_20190326_%02d.ds')

fpath = file_dir % (2, 1)
event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -1, 0, 4
freq_l, freq_h = 7, 30
decim = 1

# Make defaults
baseline = (tmin, t0)
reject = dict(mag=5e-12)

# Prepare rawobject
raw = mne.io.read_raw_ctf(fpath, preload=True)
raw.filter(freq_l, freq_h, fir_design='firwin')
picks = mne.pick_types(raw.info, meg=True, exclude='bads')

# Get events
events = mne.find_events(raw, stim_channel='UPPT001')

# Get epochs
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=decim, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=baseline,
                    reject=reject, preload=True)

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
epochs_train = epochs.copy()
labels = epochs.events[:, -1]
