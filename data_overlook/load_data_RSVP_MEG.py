# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
import sys
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
root_path = os.path.join('D:/', 'RSVP_MEG_experiment',
                         'scripts', 'data_overlook')
pdf_path = os.path.join(root_path, '..', 'results',
                        'RSVP_MEG_%s.pdf' % time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(os.path.dirname(pdf_path)):
    os.mkdir(os.path.dirname(pdf_path))
# Parameter for read raw
file_dir = os.path.join(root_path, '..', '..', 'rawdata',
                        '20190326_RSVP_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')
subject_name = 'maxuelin'
subject_idx = 2
# if len(sys.argv) > 1:
#     subject_name = sys.argv[1]
#     subject_idx = int(sys.argv[2])
run_idx = [e for e in range(4, 11)]
run_idx = run_idx[-4:]

# Parameter for preprocess raw
freq_l, freq_h = 0.1, 7
fir_design = 'firwin'
meg = True
ref_meg = False
exclude = 'bads'

# Parameter for epochs
event_id = dict(R1=1, R2=2, R3=3)
tmin, t0, tmax = -0.2, 0, 1
freq_resample = 100
decim = 10
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


def cal_events_num(events):
    label = events[:, -1]
    print(events.shape)
    [print(e, ':', sum(label == e)) for e in [1, 2, 3]]


[e.filter(freq_l, freq_h, fir_design=fir_design) for e in raw_files]

[cal_events_num(mne.find_events(e, stim_channel=stim_channel))
 for e in raw_files]

raw = mne.concatenate_raws(raw_files)
ch_names = raw.info['ch_names']
# raw.filter(freq_l, freq_h, fir_design=fir_design)
# choose channel type
picks_all_meg = mne.pick_types(
    raw.info, meg=meg, ref_meg=ref_meg, exclude=exclude)

picks = [j for j in picks_all_meg if
         ch_names[j].startswith('MLO') or
         ch_names[j].startswith('MRO') or
         ch_names[j].startswith('MRO')]

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

# Get evoked
keys = [_ for _ in event_id.keys()]
evokeds = [epochs[eid].average() for eid in keys]

figures = []

# Plot evoked
print('Plotting evoked.')
for j, e in enumerate(evokeds):
    f = e.plot(spatial_colors=True, window_title=keys[j], show=False)
    figures.append(f)
    # f = e.plot_joint(title=keys[j], show=False)
    # figures.append(f)

plt.show()
