# coding: utf-8

import os
import numpy as np

'''
# This script is to show accuracy and recall rate of RSVP odd stimuli.
# The data are EEG and MEG data, using all epochs run by server.
'''

'''
# Function: Set envrionment for the script.
# Output: root_dir, directory of project.
# Output: contain_dir, directory of this script.
'''
root_dir = os.path.join('D:\\', 'RSVP_MEG_experiment')
contain_dir = os.path.join(
    root_dir, 'scripts', 'RSVP_MVPA', 'acc_rec_allepochs_runbyserver')

data = {}
print('-'*80)
for mode_ in ['eeg', 'meg']:
    for fe_ in ['pca', 'xdawn']:
        for score_ in ['acc', 'rec']:
            idstr = '%s_%s_%s' % (mode_, fe_, score_)
            data[idstr] = np.load(os.path.join(
                contain_dir,
                'good_%s' % mode_,
                '%s_score_%s.npy' % (fe_, score_)))
            print('%s: %f' % (idstr, np.mean(data[idstr])))

print('-'*80)
for mode_ in ['eeg', 'meg']:
    for fe_ in ['pca', 'xdawn']:
        mean = sum([np.mean(data['%s_%s_%s' % (mode_, fe_, score_)])
                    for score_ in ['acc', 'rec']]) / 2
        print('%s_%s_mean: %f' % (mode_, fe_, mean))
