# code: utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os


file_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'rawdata',
                        '20190326_MI_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')

event_id = dict(MI1=1, MI2=2)
tmin, t0, tmax = -1, 0, 4
freq_l, freq_h = 1, 120
decim = 1

# Make defaults
baseline = (tmin, t0)
reject = dict(mag=5e-12)

# Prepare rawobject
raw_files = [mne.io.read_raw_ctf(
    file_dir % ('maxuelin', 2, j), preload=True) for j in [1, 2]]
raw = mne.concatenate_raws(raw_files)
raw.filter(freq_l, freq_h, fir_design='firwin')
picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')

# Get events
events = mne.find_events(raw, stim_channel='UPPT001')

# Get epochs
epochs = mne.Epochs(raw, event_id=event_id, events=events,
                    decim=decim, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=baseline,
                    reject=reject, preload=True)
epochs.resample(120, npad="auto")

ch_names = [e[0:5] for e in epochs.ch_names]
layout_all = mne.find_layout(epochs.info)
ex = [e for e in layout_all.names if e not in ch_names]
# layout_all.plot(picks=[layout_all.names.index(e) for e in ex])
layout = mne.find_layout(epochs.info, exclude=ex)

# Get evoked
evoked_1 = epochs['MI1'].average()
evoked_2 = epochs['MI2'].average()

# Plot evoked
fig, axes = plt.subplots(2, 1)
evoked_1.plot(axes=axes[0], spatial_colors=True, show=False)
evoked_2.plot(axes=axes[1], spatial_colors=True, show=False)
f1 = evoked_1.plot_joint(show=False)
f2 = evoked_2.plot_joint(show=False)

# Temporal-frequency
n_cycles = 2
freqs = np.linspace(freq_l, freq_h, 10)
freqs = np.logspace(*np.log10([freq_l, freq_h]), num=20)
power, itc = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=n_cycles,
    return_itc=True, decim=1, n_jobs=12, verbose=True)
power.plot_joint(baseline=(tmin, 0), mode='mean',
                 tmin=tmin, tmax=tmax, layout=layout,
                 show=False)
plt.show()
