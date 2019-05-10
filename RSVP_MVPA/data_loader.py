# coding: utf-8

import mne
import numpy as np
import os
import time


'''
This script is to load MEG RSVP dataset.
'''

##############
# Parameters #
##############
root_path = os.path.join('D:\\', 'RSVP_MEG_experiment')
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
prefix = 'RSVP_MEG'
