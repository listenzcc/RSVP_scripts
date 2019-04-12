# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
import time

'''
This script is a demo of loading EEG Motion Imaging data,
and plot the nature of the data into pdf file.
'''

##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
pdf_path = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                        'data_overlook', 'results',
                        'MI_EEG_%s.pdf' % time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(os.path.dirname(pdf_path)):
    os.mkdir(os.path.dirname(pdf_path))
# Parameter for read raw
file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '%s', '%s')
subject_name = '20190402_MI_EEG_maxuelin'
cnt_files = ['mxl_MI_1.cnt', 'mxl_MI_2.cnt']

# Parameter for preprocess raw
freq_l, freq_h = 7, 120
fir_design = 'firwin'
meg = False
ref_meg = False
eeg = True
exclude = 'bads'

# Parameter for epochs
event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -1, 0, 4
freq = 120
decim = 1
reject = dict()
stim_channel = 'STI 014'

# frequency
n_cycles = 2
num = 20

# multi cores
n_jobs = 12

# prepare rawobject
motage = mne.channels.read_montage('biosemi64')
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
epochs.resample(freq, npad="auto")

# Exclude abscent channels in layout
ch_names = [e[0:5] for e in epochs.ch_names]
layout_all = mne.find_layout(epochs.info)
ex = [e for e in layout_all.names if e not in ch_names]
print('Exclude channels are', ex)
# layout_all.plot(picks=[layout_all.names.index(e) for e in ex])
layout = mne.find_layout(epochs.info, exclude=ex)

# Get evoked
keys = [_ for _ in event_id.keys()]
evokeds = [epochs[eid].average() for eid in keys]

figures = []

# Plot evoked
print('Plotting evoked.')
for j, e in enumerate(evokeds):
    f = e.plot(spatial_colors=True, window_title=keys[j], show=False)
    figures.append(f)
    f = e.plot_joint(title=keys[j], show=False)
    figures.append(f)

# Calculate and plot temporal-frequency
print('Caling and ploting temporal-frequency.')
freqs = np.logspace(*np.log10([freq_l, freq_h]), num=num)
for eid in event_id.keys():
    power, itc = mne.time_frequency.tfr_morlet(
        epochs[eid], freqs=freqs, n_cycles=n_cycles,
        return_itc=True, decim=decim, n_jobs=n_jobs, verbose=True)
    f = power.plot_joint(baseline=baseline, mode='mean',
                         tmin=tmin, tmax=tmax, layout=layout,
                         title=eid, show=False)
    figures.append(f)

# Saving into pdf
print('Saving into pdf.')
with PdfPages(pdf_path) as pp:
    for f in figures:
        pp.savefig(f)

# Finally, we show figures.
plt.show()
