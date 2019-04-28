# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
import time

'''
This script is a demo of loading EEG RSVP data,
and plot the nature of the data into pdf file.
'''

##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
root_path = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                         'data_overlook')
pdf_path = os.path.join(root_path, 'results',
                        'RSVP_EEG_%s.pdf' % time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(os.path.dirname(pdf_path)):
    os.mkdir(os.path.dirname(pdf_path))
# Parameter for read raw
file_dir = os.path.join(root_path, '..', '..', 'rawdata', 'ww_RSVP',
                        '%s', 'ww_rsvp_%d.cnt')
subject_name = '20190226_RSVP_EEG_weiwei'
cnt_files_idx = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
cnt_files_idx = [10, 11, 12]

# Parameter for preprocess raw
freq_l, freq_h = 0.1, 7
fir_design = 'firwin'
meg = False
ref_meg = False
eeg = True
exclude = ['M1', 'M2', 'bads']

# Parameter for epochs
event_id = dict(R1=1, R2=2, R3=3)
tmin, t0, tmax = -1, 0, 1
freq_resample = 100
decim = 1
reject = dict()
stim_channel = 'STI 014'

# frequency
n_cycles = 2
num = 20

# multi cores
n_jobs = 12

# prepare rawobject
# motage = mne.channels.read_montage('biosemi64')
motage = mne.channels.read_montage('standard_1020')

raw_files = [mne.io.read_raw_cnt(
    file_dir % (subject_name, j), motage, preload=True) for j in cnt_files_idx]


def cal_events_num(events):
    label = events[:, -1]
    print(events.shape)
    [print(e, ':', sum(label == e)) for e in [1, 2, 3]]


[cal_events_num(mne.find_events(e, stim_channel=stim_channel))
 for e in raw_files]

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
baseline = (None, 0)
baseline = None
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=decim, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=baseline,
                    reject=reject, preload=True)
# epochs.resample(freq_resample, npad="auto")

# Exclude abscent channels in layout
ch_names = [e[0:5] for e in epochs.ch_names]
layout_all = mne.find_layout(epochs.info)
ex = [e for e in layout_all.names if e not in ch_names]
print('Exclude channels are', ex)
# layout_all.plot(picks=[layout_all.names.index(e) for e in ex])
layout = mne.find_layout(epochs.info, exclude=ex)

# Get evoked
keys = [_ for _ in event_id.keys()]
evokeds = [epochs[eid].average(method='mean') for eid in keys]

figures = []

# Plot evoked
print('Plotting evoked.')
for j, e in enumerate(evokeds):
    f = e.plot(spatial_colors=True, window_title=keys[j], show=False)
    figures.append(f)

# Finally, we show figures.
plt.show()
