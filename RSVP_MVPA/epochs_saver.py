# coding: utf-8
'''
This script is to load MEG RSVP dataset,
and save epochs as fif file.
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
'''
# file_dir = os.path.join(root_dir, 'rawdata', '20190326_RSVP_MEG_%s',
#                         'S%02d_lixiangTHU_20190326_%02d.ds')  # MEG
file_dir = os.path.join(root_dir, 'rawdata', 'ww_RSVP',
                        '20190228_RSVP_EEG_weiwei', 'ww_rsvp_%d.cnt')  # EEG
freq_l, freq_h = 0.1, 7  # seconds
fir_design = 'firwin'

'''
# Function: Preparing motage for EEG dataset
# Output: motage, motage of standard_1020
'''
motage = mne.channels.read_montage('standard_1020')

# for run_idx in range(4, 11):  # MEG
for run_idx in range(1, 20):  # EEG
    print(run_idx)
    if run_idx in [1, 6, 18]:
        continue

    # raw = mne.io.read_raw_ctf(file_dir % (subject_name, subject_idx, run_idx),
    #                           preload=True)  # MEG
    raw = mne.io.read_raw_cnt(file_dir % (
        run_idx), motage, preload=True)  # EEG
    raw = raw.filter(freq_l, freq_h, fir_design=fir_design)

    '''
    # Function: Get epochs.
    # Output: epochs, resampled epochs of norm and odd stimuli.
    '''
    # picks = mne.pick_types(
    #     raw.info, meg=True, ref_meg=False, exclude='bads')  # MEG
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')  # EEG
    # stim_channel = 'UPPT001'  # MEG
    stim_channel = 'STI 014'  # EEG
    events = mne.find_events(raw, stim_channel=stim_channel)
    event_id = dict(odd=1, norm=2)
    tmin, t0, tmax = -0.2, 0, 1  # Seconds
    freq_resample = 200  # Hz
    reject = dict()

    epochs = mne.Epochs(raw, picks=picks, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=(tmin, t0), detrend=1,
                        reject=reject, preload=True)
    epochs.resample(freq_resample, npad='auto')

    '''
    # Function: Save epochs.
    # Output: None.
    '''
    epochs.save(os.path.join(results_dir, 'eeg_ww_epochs_%d-epo.fif' %
                             run_idx), verbose=True)
