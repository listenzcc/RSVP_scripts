# coding: utf-8
'''
This script is to load MEG RSVP dataset.
'''

import mne
import os
import time

'''
# Function: Setting evrionment for the script.
# Output: root_path, directory of project.
# Output: time_stamp, string of beginning time of the script.
# Output: id_string, customer identifier string.
# Output: results_dir, directory for storing results.
'''
root_dir = os.path.join('D:\\', 'RSVP_MEG_experiment')
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
id_string = 'RSVP_MEG'
results_dir = os.path.join(root_dir, 'scripts', 'RSVP_MVPA', 'results')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

'''
# Function: Reading raw_files.
# Output: raw_files, raw files in list, every element for a run.
# Output: info, measurement information.
'''
file_dir = os.path.join(root_dir, 'rawdata', '20190326_RSVP_MEG_%s',
                        'S%02d_lixiangTHU_20190326_%02d.ds')
subject_name = 'maxuelin'
subject_idx = 2
run_idx = [e for e in range(4, 11)]
run_idx = [5, 7]

raw_files = [mne.io.read_raw_ctf(
    file_dir % (subject_name, subject_idx, j), preload=True)
    for j in run_idx]
info = raw_files[0].info

'''
# Function: Filter and concatenate raw_files.
# Output: raw, concatenated raw_files.
'''
freq_l, freq_h = 0.1, 7  # seconds
fir_design = 'firwin'

raw = mne.concatenate_raws(
    [e.filter(freq_l, freq_h, fir_design=fir_design) for e in raw_files])

'''
# Function: Get epochs.
# Output: epochs, resampled epochs of norm and odd stimuli.
'''
picks = mne.pick_types(info, meg=True, ref_meg=False, exclude='bads')
stim_channel = 'UPPT001'
events = mne.find_events(raw, stim_channel=stim_channel)
event_id = dict(odd=1, norm=2)
tmin, t0, tmax = -0.2, 0, 1  # Seconds
freq_resample = 200  # Hz
reject = dict(mag=5e-12)

epochs = mne.Epochs(raw, picks=picks, events=events, event_id=event_id,
                    tmin=tmin, tmax=tmax, baseline=(tmin, t0), detrend=1,
                    reject=reject, preload=True)
epochs.resample(freq_resample, npad='auto')

'''
# Function: Save epochs.
# Output: None.
'''
epochs.save(os.path.join(results_dir, 'epochs-epo.fif'), verbose=True)
