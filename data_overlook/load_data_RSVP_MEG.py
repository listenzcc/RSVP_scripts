# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
import time

'''
This script is a demo of loading MEG RSVP data,
and plot the nature of the data into pdf file.
'''

##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
pdf_path = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                        'data_overlook', 'results',
                        'RSVP_MEG_%s.pdf' % time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(os.path.dirname(pdf_path)):
    os.mkdir(os.path.dirname(pdf_path))
# Parameter for read raw
file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '20190326_RSVP_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')
subject_name = 'maxuelin'
subject_idx = 2
run_idx = [e for e in range(4, 12)]
run_idx = run_idx[-4:]

# Parameter for preprocess raw
freq_l, freq_h = 7, 360
fir_design = 'firwin'
meg = True
ref_meg = False
exclude = 'bads'

# Parameter for epochs
event_id = dict(R1=1, R2=2, R3=3)
tmin, t0, tmax = -1, 0, 4
freq = 360
decim = 1
reject = dict(mag=5e-12)
stim_channel = 'UPPT001'

# frequency
n_cycles = 2
num = 20

# multi cores
n_jobs = 12

# prepare rawobject
raw_files = [mne.io.read_raw_ctf(
    file_dir % (subject_name, subject_idx, j), preload=True) for j in run_idx]
# 10 runs in each raw_file,
# 14 blocks in each run,
# 10 pictures in each block,
# including 2 target pictures.


def cal_events_nun(events):
    label = events[:, -1]
    print(events.shape)
    [print(e, ':', sum(label == e)) for e in [1, 2, 3]]


[cal_events_nun(mne.find_events(e, stim_channel=stim_channel))
 for e in raw_files]

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
